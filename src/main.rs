use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_CONFIG: &str = r#"# Slimefinder 配置 / Configuration

[mask]
# 世界种子 / World seed
world_seed = 0
# 是否启用消失球体（>128m 将被排除）/ Enable despawn sphere (>128m trimmed)
despawn_sphere = true
# 是否启用玩家安全区 / Enable player exclusion sphere
exclusion_sphere = true
# 玩家 Y 轴相对高度 / Player Y-offset
y_offset = 0
# 判断区块在遮罩内所需的最少方块数 / Minimum block weight per chunk
chunk_weight = 0

[search]
# 搜索中心（格式: chunkXcInX,chunkZcInZ）/ search center
center_pos = "0c0,0c0"
# 内部需要跳过的正方形宽度 / inner square width to skip
min_width = 0
# 搜索正方形宽度（max_width^2 个区块）/ outer square width
max_width = 1
# 是否扫描 Chunk 内全部 256 个方块 / iterate every in-chunk position
fine_search = false
# 匹配所需的最小/最大方块数量 / min & max block counts
min_block_size = 0
max_block_size = 73984
# 匹配所需的最小/最大区块数量 / min & max chunk counts
min_chunk_size = 0
max_chunk_size = 289
# 输出文件 / output CSV file
output_file = "results.csv"
# 是否追加写入 / append instead of overwrite
append = false
"#;

const R_CHUNK: i32 = 8;
const MASK_WIDTH: usize = R_CHUNK as usize * 2 + 1;
const BLOCKS_PER_CHUNK: i32 = 16;
const BLOCKS_PER_CHUNK_USIZE: usize = 16;
const MAX_BLOCK_SURFACE: u32 =
    (MASK_WIDTH * MASK_WIDTH * BLOCKS_PER_CHUNK_USIZE * BLOCKS_PER_CHUNK_USIZE) as u32;
const MAX_CHUNK_SURFACE: u32 = (MASK_WIDTH * MASK_WIDTH) as u32;
const CHUNK_BATCH_SIZE: usize = 2048;
const CSV_HEADER: &str = "block-position;chunk-position;blockSize;chunkSize";
const PROGRESS_REFRESH_MS: u64 = 350;
const RESULT_FLUSH_THRESHOLD: usize = 1_024_000;

fn main() -> Result<()> {
    let args = CliArgs::parse()?;
    if !ensure_config_exists(&args.config_path)? {
        println!(
            "已生成默认配置文件，请编辑后重新运行。\nDefault config created at {}. Please edit it and rerun.",
            args.config_path.display()
        );
        return Ok(());
    }

    let config_text = fs::read_to_string(&args.config_path).with_context(|| {
        format!(
            "无法读取配置 / Failed to read {}",
            args.config_path.display()
        )
    })?;
    let config: AppConfig = toml::from_str(&config_text).with_context(|| {
        format!(
            "无法解析配置 / Failed to parse {}",
            args.config_path.display()
        )
    })?;
    run_search(&config, &args.config_path)
}

fn run_search(config: &AppConfig, config_path: &Path) -> Result<()> {
    let mut path = SearchPath::new(
        config.search.center_pos.chunk,
        config.search.min_width as i32,
        config.search.max_width as i32,
    );
    let chunk_total = path.path_length();
    if chunk_total == 0 {
        println!(
            "max_width 必须大于 min_width，当前没有可搜索的区块。\nNo positions were enqueued because max_width <= min_width."
        );
        return Ok(());
    }
    let in_points = build_in_points(&config.search);
    let total = chunk_total * in_points.len() as u64;
    if total == 0 {
        println!("未创建任何任务 / No jobs were scheduled.");
        return Ok(());
    }

    println!(
        "搜索条件 / Criteria: blockSize ∈ [{},{}] 或 or chunkSize ∈ [{},{}]",
        config.search.min_block_size,
        config.search.max_block_size,
        config.search.min_chunk_size,
        config.search.max_chunk_size
    );
    println!(
        "结果文件 / Output file: {} (append={})",
        config.search.output_file, config.search.append
    );

    let mask_cfg = Arc::new(config.mask.clone());
    let template_cache = Arc::new(build_template_cache(&in_points, mask_cfg.as_ref()));
    let bounds = Arc::new(SearchBounds::from(&config.search));
    let matches_counter = Arc::new(AtomicU64::new(0));
    let completed = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let done = Arc::new(AtomicBool::new(false));

    let progress_handle = {
        let total = total;
        let matches_counter = Arc::clone(&matches_counter);
        let completed = Arc::clone(&completed);
        let done = Arc::clone(&done);
        thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                let line = format_progress_line(
                    total,
                    completed.load(Ordering::Relaxed),
                    matches_counter.load(Ordering::Relaxed),
                    start,
                );
                print!("\r{}", line);
                let _ = io::stdout().flush();
                thread::sleep(Duration::from_millis(PROGRESS_REFRESH_MS));
            }
            let line = format_progress_line(
                total,
                completed.load(Ordering::Relaxed),
                matches_counter.load(Ordering::Relaxed),
                start,
            );
            println!("\r{}", line);
        })
    };

    let output_path = resolve_output_path(config_path, &config.search.output_file);
    let mut result_writer = ResultWriter::new(&output_path, config.search.append)?;
    let mut total_matches_written = 0usize;
    let mut extrema = Aggregation::default();
    let mut pending_matches = Vec::with_capacity(RESULT_FLUSH_THRESHOLD.min(1024));

    // 分批生成任务，避免在巨大范围内一次性分配全部坐标。
    // Batch the jobs so extremely wide searches do not allocate enormous vectors.
    let mut processed_chunks = 0u64;
    let mut chunk_batch = Vec::with_capacity(CHUNK_BATCH_SIZE);
    let mut job_batch = Vec::with_capacity(CHUNK_BATCH_SIZE * in_points.len());
    loop {
        fill_chunk_batch(&mut path, &mut chunk_batch, CHUNK_BATCH_SIZE);
        if chunk_batch.is_empty() {
            break;
        }
        let remaining_chunks = chunk_total.saturating_sub(processed_chunks);
        if remaining_chunks == 0 {
            break;
        }
        if chunk_batch.len() as u64 > remaining_chunks {
            chunk_batch.truncate(remaining_chunks as usize);
        }
        if chunk_batch.is_empty() {
            break;
        }
        processed_chunks += chunk_batch.len() as u64;
        job_batch.clear();
        for chunk in &chunk_batch {
            for in_point in &in_points {
                job_batch.push(MaskJob {
                    chunk: *chunk,
                    in_block: *in_point,
                });
            }
        }

        let mut batch = job_batch
            .par_iter()
            .fold(
                || Aggregation::default(),
                |mut agg, job| {
                    let template = template_cache
                        .get(&job.in_block)
                        .expect("missing precomputed mask template");
                    let data = compute_mask(mask_cfg.as_ref(), template, job);
                    let is_match = bounds.matches(&data);
                    if is_match {
                        matches_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    completed.fetch_add(1, Ordering::Relaxed);
                    agg.add_result(data, is_match);
                    agg
                },
            )
            .reduce(Aggregation::default, Aggregation::merge);
        batch.matches.sort_by(|a, b| {
            (a.chunk.x, a.chunk.z, a.in_block.x, a.in_block.z).cmp(&(
                b.chunk.x,
                b.chunk.z,
                b.in_block.x,
                b.in_block.z,
            ))
        });
        pending_matches.append(&mut batch.matches);
        if pending_matches.len() >= RESULT_FLUSH_THRESHOLD {
            result_writer.write_batch(&pending_matches)?;
            total_matches_written += pending_matches.len();
            pending_matches.clear();
        }
        extrema.absorb_extrema(&batch);
        if processed_chunks >= chunk_total {
            break;
        }
    }

    done.store(true, Ordering::Relaxed);
    let _ = progress_handle.join();

    if !pending_matches.is_empty() {
        result_writer.write_batch(&pending_matches)?;
        total_matches_written += pending_matches.len();
        pending_matches.clear();
    }
    result_writer.flush()?;
    println!(
        "已写入匹配结果 / Matches written: {} → {}",
        total_matches_written,
        output_path.display()
    );

    if total > 0 {
        let elapsed = start.elapsed();
        let nanos_per = elapsed.as_nanos() / total as u128;
        println!("平均耗时 / Avg ns per position: {}", nanos_per);
    }

    if let Some(min_block) = extrema.min_block {
        println!(
            "最小 block / Smallest block cluster: {} @ {} ({})",
            min_block.block_ratio(),
            min_block.chunk_string(),
            min_block.block_string()
        );
    }
    if let Some(max_block) = extrema.max_block {
        println!(
            "最大 block / Largest block cluster: {} @ {} ({})",
            max_block.block_ratio(),
            max_block.chunk_string(),
            max_block.block_string()
        );
    }
    if let Some(min_chunk) = extrema.min_chunk {
        println!(
            "最小 chunk / Smallest chunk cluster: {} @ {} ({})",
            min_chunk.chunk_ratio(),
            min_chunk.chunk_string(),
            min_chunk.block_string()
        );
    }
    if let Some(max_chunk) = extrema.max_chunk {
        println!(
            "最大 chunk / Largest chunk cluster: {} @ {} ({})",
            max_chunk.chunk_ratio(),
            max_chunk.chunk_string(),
            max_chunk.block_string()
        );
    }
    Ok(())
}

fn ensure_config_exists(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(true);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("无法创建目录 / Failed to create {}", parent.display()))?;
        }
    }
    fs::write(path, DEFAULT_CONFIG)
        .with_context(|| format!("无法写入配置 / Failed to write {}", path.display()))?;
    Ok(false)
}

#[derive(Debug)]
struct CliArgs {
    config_path: PathBuf,
}

impl CliArgs {
    fn parse() -> Result<Self> {
        let mut args = env::args().skip(1);
        let mut config_path = PathBuf::from("slimefinder.toml");
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-c" | "--config" => {
                    let value = args.next().ok_or_else(|| {
                        anyhow!("--config 需要文件路径 / missing path after --config")
                    })?;
                    config_path = PathBuf::from(value);
                }
                "-h" | "--help" => {
                    print_usage();
                    process::exit(0);
                }
                unknown => {
                    return Err(anyhow!("未知参数 / Unknown argument: {}", unknown));
                }
            }
        }
        Ok(Self { config_path })
    }
}

fn print_usage() {
    println!(
        "Slimefinder (Rust)\n\
        用法 Usage:\n  slimefinder [--config slimefinder.toml]\n\n\
        --config, -c  指定配置文件 / specify config file\n\
        --help, -h    显示帮助 / show help"
    );
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(default)]
    mask: MaskConfig,
    #[serde(default)]
    search: SearchConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct MaskConfig {
    world_seed: i64,
    despawn_sphere: bool,
    exclusion_sphere: bool,
    y_offset: i32,
    chunk_weight: u32,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            world_seed: 0,
            despawn_sphere: true,
            exclusion_sphere: true,
            y_offset: 0,
            chunk_weight: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct SearchConfig {
    center_pos: Position,
    min_width: u32,
    max_width: u32,
    fine_search: bool,
    min_block_size: u32,
    max_block_size: u32,
    min_chunk_size: u32,
    max_chunk_size: u32,
    output_file: String,
    append: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            center_pos: Position::default(),
            min_width: 0,
            max_width: 1,
            fine_search: false,
            min_block_size: 0,
            max_block_size: MAX_BLOCK_SURFACE,
            min_chunk_size: 0,
            max_chunk_size: MAX_CHUNK_SURFACE,
            output_file: "results.csv".to_string(),
            append: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Point {
    x: i32,
    z: i32,
}

impl Point {
    fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    fn move_by(&mut self, count: i32, direction: Direction) {
        self.x += direction.dx() * count;
        self.z += direction.dz() * count;
    }
}

#[derive(Clone, Copy, Debug)]
struct Position {
    chunk: Point,
    in_block: Point,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            chunk: Point::new(0, 0),
            in_block: Point::new(0, 0),
        }
    }
}

impl Position {
    fn from_block(block_x: i32, block_z: i32) -> Self {
        let chunk_x = block_x.div_euclid(BLOCKS_PER_CHUNK);
        let chunk_z = block_z.div_euclid(BLOCKS_PER_CHUNK);
        let in_x = block_x.rem_euclid(BLOCKS_PER_CHUNK);
        let in_z = block_z.rem_euclid(BLOCKS_PER_CHUNK);
        Self {
            chunk: Point::new(chunk_x, chunk_z),
            in_block: Point::new(in_x, in_z),
        }
    }

    fn from_chunk(chunk_x: i32, chunk_z: i32, in_x: i32, in_z: i32) -> Self {
        Self {
            chunk: Point::new(chunk_x, chunk_z),
            in_block: Point::new(in_x, in_z),
        }
    }
}

impl<'de> Deserialize<'de> for Position {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        parse_position(&text).map_err(serde::de::Error::custom)
    }
}

fn parse_position(raw: &str) -> Result<Position, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("位置字符串为空 / empty position string".into());
    }
    let parts: Vec<&str> = trimmed.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err("坐标格式错误 / invalid coordinate format".into());
    }
    let x_parts: Vec<&str> = parts[0].split('c').collect();
    let z_parts: Vec<&str> = parts[1].split('c').collect();
    match (x_parts.len(), z_parts.len()) {
        (1, 1) => {
            let block_x = x_parts[0]
                .parse::<i32>()
                .map_err(|_| "X 坐标无效 / invalid X block coordinate".to_string())?;
            let block_z = z_parts[0]
                .parse::<i32>()
                .map_err(|_| "Z 坐标无效 / invalid Z block coordinate".to_string())?;
            Ok(Position::from_block(block_x, block_z))
        }
        (2, 2) => {
            let chunk_x = x_parts[0]
                .parse::<i32>()
                .map_err(|_| "X chunk 无效 / invalid X chunk value".to_string())?;
            let in_x = x_parts[1]
                .parse::<i32>()
                .map_err(|_| "X in-chunk 无效 / invalid X in-chunk value".to_string())?;
            let chunk_z = z_parts[0]
                .parse::<i32>()
                .map_err(|_| "Z chunk 无效 / invalid Z chunk value".to_string())?;
            let in_z = z_parts[1]
                .parse::<i32>()
                .map_err(|_| "Z in-chunk 无效 / invalid Z in-chunk value".to_string())?;
            Ok(Position::from_chunk(chunk_x, chunk_z, in_x, in_z))
        }
        _ => Err("坐标需要 block 或 chunkcIn 格式 / invalid format".into()),
    }
}

#[derive(Debug, Clone)]
struct SearchBounds {
    min_block_size: u32,
    max_block_size: u32,
    min_chunk_size: u32,
    max_chunk_size: u32,
}

impl SearchBounds {
    fn from(cfg: &SearchConfig) -> Self {
        Self {
            min_block_size: cfg.min_block_size,
            max_block_size: cfg.max_block_size,
            min_chunk_size: cfg.min_chunk_size,
            max_chunk_size: cfg.max_chunk_size,
        }
    }

    fn matches(&self, data: &MaskData) -> bool {
        let block_ok =
            data.block_size >= self.min_block_size && data.block_size <= self.max_block_size;
        let chunk_ok =
            data.chunk_size >= self.min_chunk_size && data.chunk_size <= self.max_chunk_size;
        block_ok || chunk_ok
    }
}

#[derive(Clone, Debug)]
struct MaskTemplate {
    chunk_weights: [[u32; MASK_WIDTH]; MASK_WIDTH],
    block_surface_area: u32,
    chunk_surface_area: u32,
}

#[derive(Clone, Copy, Debug)]
struct MaskGeometry {
    r_exclusion_sq: i64,
    r_despawn_sq: i64,
}

impl MaskGeometry {
    fn from_config(cfg: &MaskConfig) -> Self {
        let y = cfg.y_offset as i64;
        let y_sq = y * y;
        let exclusion_limit = 24_i64 * 24_i64;
        let r_exclusion_sq = exclusion_limit - y_sq.min(exclusion_limit);
        let r_despawn_sq = 128_i64 * 128_i64 - y_sq;
        Self {
            r_exclusion_sq,
            r_despawn_sq,
        }
    }
}

fn build_template_cache(points: &[Point], cfg: &MaskConfig) -> HashMap<Point, MaskTemplate> {
    let geometry = MaskGeometry::from_config(cfg);
    let mut map = HashMap::new();
    for point in points {
        map.entry(*point)
            .or_insert_with(|| build_template(cfg, &geometry, *point));
    }
    map
}

fn build_template(cfg: &MaskConfig, geometry: &MaskGeometry, in_block: Point) -> MaskTemplate {
    let mut chunk_weights = [[0u32; MASK_WIDTH]; MASK_WIDTH];
    let mut block_surface_area = 0u32;
    let mut chunk_surface_area = 0u32;
    for (dx_idx, chunk_x) in (-R_CHUNK..=R_CHUNK).enumerate() {
        for (dz_idx, chunk_z) in (-R_CHUNK..=R_CHUNK).enumerate() {
            let mut weight = 0u32;
            for local_x in 0..BLOCKS_PER_CHUNK {
                for local_z in 0..BLOCKS_PER_CHUNK {
                    let block_x = chunk_x * BLOCKS_PER_CHUNK + local_x;
                    let block_z = chunk_z * BLOCKS_PER_CHUNK + local_z;
                    if is_block_inside(cfg, geometry, in_block, block_x, block_z) {
                        weight += 1;
                    }
                }
            }
            chunk_weights[dx_idx][dz_idx] = weight;
            block_surface_area += weight;
            if weight > cfg.chunk_weight {
                chunk_surface_area += 1;
            }
        }
    }
    MaskTemplate {
        chunk_weights,
        block_surface_area,
        chunk_surface_area,
    }
}

fn is_block_inside(
    cfg: &MaskConfig,
    geometry: &MaskGeometry,
    in_block: Point,
    block_x: i32,
    block_z: i32,
) -> bool {
    let dx = block_x - in_block.x;
    let dz = block_z - in_block.z;
    let dsqr = (dx as i64) * (dx as i64) + (dz as i64) * (dz as i64);
    if cfg.despawn_sphere && dsqr > geometry.r_despawn_sq {
        return false;
    }
    if cfg.exclusion_sphere && dsqr <= geometry.r_exclusion_sq {
        return false;
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MaskJob {
    chunk: Point,
    in_block: Point,
}

fn build_in_points(cfg: &SearchConfig) -> Vec<Point> {
    if cfg.fine_search {
        let mut v = Vec::with_capacity(256);
        for x in 0..BLOCKS_PER_CHUNK {
            for z in 0..BLOCKS_PER_CHUNK {
                v.push(Point::new(x, z));
            }
        }
        v
    } else {
        vec![cfg.center_pos.in_block]
    }
}

fn fill_chunk_batch(path: &mut SearchPath, buffer: &mut Vec<Point>, limit: usize) {
    buffer.clear();
    for _ in 0..limit {
        match path.next() {
            Some(point) => buffer.push(point),
            None => break,
        }
    }
}

fn compute_mask(cfg: &MaskConfig, template: &MaskTemplate, job: &MaskJob) -> MaskData {
    let mut block_size = 0u32;
    let mut chunk_size = 0u32;
    for (dx_idx, row) in template.chunk_weights.iter().enumerate() {
        let rel_x = dx_idx as i32 - R_CHUNK;
        for (dz_idx, &weight) in row.iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let rel_z = dz_idx as i32 - R_CHUNK;
            if is_slime_chunk(cfg.world_seed, job.chunk.x + rel_x, job.chunk.z + rel_z) {
                block_size += weight;
                if weight > cfg.chunk_weight {
                    chunk_size += 1;
                }
            }
        }
    }
    MaskData {
        chunk: job.chunk,
        in_block: job.in_block,
        block_surface_area: template.block_surface_area,
        chunk_surface_area: template.chunk_surface_area,
        block_size,
        chunk_size,
    }
}

#[derive(Clone, Debug)]
struct MaskData {
    chunk: Point,
    in_block: Point,
    block_surface_area: u32,
    chunk_surface_area: u32,
    block_size: u32,
    chunk_size: u32,
}

impl MaskData {
    fn chunk_string(&self) -> String {
        format!(
            "{}c{},{}c{}",
            self.chunk.x, self.in_block.x, self.chunk.z, self.in_block.z
        )
    }

    fn block_string(&self) -> String {
        let block_x = self.chunk.x * BLOCKS_PER_CHUNK + self.in_block.x;
        let block_z = self.chunk.z * BLOCKS_PER_CHUNK + self.in_block.z;
        format!("{},{}", block_x, block_z)
    }

    fn block_ratio(&self) -> String {
        format!("{}/{}", self.block_size, self.block_surface_area)
    }

    fn chunk_ratio(&self) -> String {
        format!("{}/{}", self.chunk_size, self.chunk_surface_area)
    }
}

#[derive(Default)]
struct Aggregation {
    matches: Vec<MaskData>,
    max_block: Option<MaskData>,
    min_block: Option<MaskData>,
    max_chunk: Option<MaskData>,
    min_chunk: Option<MaskData>,
}

impl Aggregation {
    fn add_result(&mut self, data: MaskData, is_match: bool) {
        self.update_extrema(data.clone());
        if is_match {
            self.matches.push(data);
        }
    }

    fn update_extrema(&mut self, data: MaskData) {
        update_slot(&mut self.max_block, data.clone(), |cand, curr| {
            cand.block_size > curr.block_size
        });
        update_slot(&mut self.min_block, data.clone(), |cand, curr| {
            cand.block_size < curr.block_size
        });
        update_slot(&mut self.max_chunk, data.clone(), |cand, curr| {
            cand.chunk_size > curr.chunk_size
        });
        update_slot(&mut self.min_chunk, data, |cand, curr| {
            cand.chunk_size < curr.chunk_size
        });
    }

    fn merge(mut self, mut other: Aggregation) -> Aggregation {
        self.matches.append(&mut other.matches);
        adopt_extremum(&mut self.max_block, other.max_block, |cand, curr| {
            cand.block_size > curr.block_size
        });
        adopt_extremum(&mut self.min_block, other.min_block, |cand, curr| {
            cand.block_size < curr.block_size
        });
        adopt_extremum(&mut self.max_chunk, other.max_chunk, |cand, curr| {
            cand.chunk_size > curr.chunk_size
        });
        adopt_extremum(&mut self.min_chunk, other.min_chunk, |cand, curr| {
            cand.chunk_size < curr.chunk_size
        });
        self
    }

    fn absorb_extrema(&mut self, other: &Aggregation) {
        if let Some(ref candidate) = other.max_block {
            update_slot(&mut self.max_block, candidate.clone(), |cand, curr| {
                cand.block_size > curr.block_size
            });
        }
        if let Some(ref candidate) = other.min_block {
            update_slot(&mut self.min_block, candidate.clone(), |cand, curr| {
                cand.block_size < curr.block_size
            });
        }
        if let Some(ref candidate) = other.max_chunk {
            update_slot(&mut self.max_chunk, candidate.clone(), |cand, curr| {
                cand.chunk_size > curr.chunk_size
            });
        }
        if let Some(ref candidate) = other.min_chunk {
            update_slot(&mut self.min_chunk, candidate.clone(), |cand, curr| {
                cand.chunk_size < curr.chunk_size
            });
        }
    }
}

fn update_slot<F>(slot: &mut Option<MaskData>, data: MaskData, better: F)
where
    F: Fn(&MaskData, &MaskData) -> bool,
{
    match slot {
        Some(current) => {
            if better(&data, current) {
                *current = data;
            }
        }
        None => *slot = Some(data),
    }
}

fn adopt_extremum<F>(slot: &mut Option<MaskData>, other: Option<MaskData>, better: F)
where
    F: Fn(&MaskData, &MaskData) -> bool,
{
    if let Some(candidate) = other {
        match slot {
            Some(current) => {
                if better(&candidate, current) {
                    *slot = Some(candidate);
                }
            }
            None => *slot = Some(candidate),
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    fn turn_clockwise(self) -> Self {
        match self {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
        }
    }

    fn dx(self) -> i32 {
        match self {
            Direction::East => 1,
            Direction::West => -1,
            _ => 0,
        }
    }

    fn dz(self) -> i32 {
        match self {
            Direction::South => 1,
            Direction::North => -1,
            _ => 0,
        }
    }
}

struct SearchPath {
    center: Point,
    min_width: i32,
    max_width: i32,
    edge: i32,
    steps: i32,
    turns: i32,
    dir: Direction,
    point: Point,
    in_progress: bool,
}

impl SearchPath {
    fn new(center: Point, min_width: i32, max_width: i32) -> Self {
        let min_width = min_width.max(0);
        let max_width = max_width.max(0);
        let mut path = Self {
            center,
            min_width,
            max_width,
            edge: 0,
            steps: 0,
            turns: 0,
            dir: Direction::East,
            point: center,
            in_progress: false,
        };
        path.init();
        path
    }

    fn path_length(&self) -> u64 {
        if self.max_width <= self.min_width {
            0
        } else {
            (self.max_width as i64 * self.max_width as i64
                - self.min_width as i64 * self.min_width as i64) as u64
        }
    }

    fn init(&mut self) {
        self.steps = 0;
        if self.min_width <= 0 {
            self.dir = Direction::East;
            self.edge = self.min_width + 1;
            self.turns = 0;
            self.point = self.center;
        } else if self.min_width % 2 == 0 {
            self.dir = Direction::North;
            self.edge = self.min_width;
            self.turns = 1;
            self.point = Point::new(
                self.center.x - self.min_width / 2,
                self.center.z + self.min_width / 2,
            );
        } else {
            self.dir = Direction::South;
            self.edge = self.min_width;
            self.turns = 1;
            self.point = Point::new(
                self.center.x + (self.min_width + 1) / 2,
                self.center.z - (self.min_width - 1) / 2,
            );
        }
        self.in_progress = false;
    }

    fn step(&mut self) -> bool {
        if !self.in_progress {
            self.in_progress = true;
            return self.max_width > self.min_width;
        }
        self.point.move_by(1, self.dir);
        self.steps += 1;
        if self.edge >= self.max_width && self.steps >= self.edge {
            self.init();
        } else if self.steps >= self.edge {
            self.steps = 0;
            self.dir = self.dir.turn_clockwise();
            self.turns += 1;
            if self.turns > 1 {
                self.turns = 0;
                self.edge += 1;
            }
        }
        self.in_progress
    }

    fn point(&self) -> Option<Point> {
        if self.in_progress {
            Some(self.point)
        } else {
            None
        }
    }
}

impl Iterator for SearchPath {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step() {
            self.point()
        } else {
            None
        }
    }
}

fn resolve_output_path(config_path: &Path, output: &str) -> PathBuf {
    let out_path = Path::new(output);
    if out_path.is_absolute() {
        out_path.to_path_buf()
    } else {
        config_path
            .parent()
            .map(|p| {
                if p.as_os_str().is_empty() {
                    Path::new(".").to_path_buf()
                } else {
                    p.to_path_buf()
                }
            })
            .unwrap_or_else(|| PathBuf::from("."))
            .join(out_path)
    }
}

struct ResultWriter {
    path: PathBuf,
    writer: BufWriter<std::fs::File>,
}

impl ResultWriter {
    fn new(path: &Path, append: bool) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("无法创建目录 / Failed to create {}", parent.display())
                })?;
            }
        }
        let need_header = if append {
            fs::metadata(path)
                .map(|meta| meta.len() == 0)
                .unwrap_or(true)
        } else {
            true
        };
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .with_context(|| format!("无法打开结果文件 / Failed to open {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        if need_header {
            writeln!(writer, "{}", CSV_HEADER)?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            writer,
        })
    }

    fn write_batch(&mut self, matches: &[MaskData]) -> Result<()> {
        for data in matches {
            writeln!(
                self.writer,
                "{};{};{}/{};{}/{}",
                data.block_string(),
                data.chunk_string(),
                data.block_size,
                data.block_surface_area,
                data.chunk_size,
                data.chunk_surface_area
            )?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().with_context(|| {
            format!(
                "无法刷新结果文件 / Failed to flush {}",
                self.path.display()
            )
        })
    }
}

fn format_progress_line(total: u64, completed: u64, matches: u64, start: Instant) -> String {
    if total == 0 {
        return "尚未开始 / No work scheduled".to_string();
    }
    let capped_completed = completed.min(total);
    let progress = capped_completed as f64 / total as f64;
    let elapsed = start.elapsed();
    let remaining = if completed > 0 {
        let secs_per = elapsed.as_secs_f64() / completed as f64;
        Duration::from_secs_f64(total.saturating_sub(completed) as f64 * secs_per)
    } else {
        Duration::from_secs(0)
    };
    let speed = if elapsed.as_secs_f64() > 0.0 {
        completed as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let remaining_str = if completed > 0 {
        format_duration(remaining)
    } else {
        "--:--:--".to_string()
    };
    format!(
        "{:>6.2}% | 速度 speed: {:>10.2} pos/s | 匹配 matches: {} | 进度 progress: {}/{} | 已用 elapsed: {} | 剩余 remaining: {}",
        progress * 100.0,
        speed,
        matches,
        capped_completed,
        total,
        format_duration(elapsed),
        remaining_str
    )
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, seconds)
}

struct JavaRandom {
    seed: i64,
}

impl JavaRandom {
    const MULTIPLIER: i64 = 0x5DEECE66D;
    const ADDEND: i64 = 0xB;
    const MASK: i64 = (1_i64 << 48) - 1;

    fn new(seed: i64) -> Self {
        let initial = (seed ^ Self::MULTIPLIER) & Self::MASK;
        Self { seed: initial }
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.seed = (self
            .seed
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND))
            & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    fn next_int(&mut self, bound: i32) -> i32 {
        if bound <= 0 {
            panic!("bound must be positive");
        }
        if bound & (bound - 1) == 0 {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let value = bits % bound;
            if bits - value + (bound - 1) >= 0 {
                return value;
            }
        }
    }
}

fn is_slime_chunk(seed: i64, chunk_x: i32, chunk_z: i32) -> bool {
    // The Java implementation performs these multiplications in 32-bit ints and then
    // casts to long, so we replicate that overflow behaviour explicitly.
    let chunk_x_sq = chunk_x.wrapping_mul(chunk_x);
    let chunk_z_sq = chunk_z.wrapping_mul(chunk_z);
    let term_x_sq = i64::from(chunk_x_sq.wrapping_mul(4_987_142));
    let term_x = i64::from(chunk_x.wrapping_mul(5_947_611));
    let term_z_sq = i64::from(chunk_z_sq).wrapping_mul(4_392_871_i64);
    let term_z = i64::from(chunk_z.wrapping_mul(389_711));
    let mixed_seed = seed
        .wrapping_add(term_x_sq)
        .wrapping_add(term_x)
        .wrapping_add(term_z_sq)
        .wrapping_add(term_z)
        ^ 987_234_911_i64;
    let mut random = JavaRandom::new(mixed_seed);
    random.next_int(10) == 0
}
