# Slimefinder 用户手册 / User Manual

> Rust + Rayon 版本  
> Rust + Rayon Edition

Slimefinder 现在是一个使用 Rust 编写的命令行工具，通过 Rayon 并行遍历 Minecraft 世界中的区块，寻找满足特定史莱姆区块数量的区域。程序使用 `slimefinder.toml` 进行配置，所有参数均提供中英文说明，便于快速定制。

## 运行 / Usage

```bash
cargo run --release -- --config slimefinder.toml
```

- 第一次运行若未找到配置文件，会自动生成带有注释的 `slimefinder.toml` 并退出，方便先编辑。  
  When no config file is present, a default `slimefinder.toml` with comments is created and the program exits so you can edit it.
- 支持的命令行参数：  
  Supported CLI flags:
  - `--config, -c <path>`：指定配置文件路径 / custom config path  
  - `--help, -h`：显示帮助信息 / show help text

程序会依据配置在终端实时展示进度、匹配数量和预计剩余时间，并在完成后写出 `results.csv`（可配置路径）。CSV 使用 `;` 作为分隔符，首行包含标题。

## 配置结构 / Configuration Layout

配置文件是一个标准 TOML，分为 `[mask]` 和 `[search]` 两个部分。下面的表格列出主要字段及默认值。

### `[mask]`

| 字段 Field | 类型 Type | 默认 Default | 说明 Description |
| --- | --- | --- | --- |
| `world_seed` | i64 | `0` | 世界种子 / Minecraft world seed |
| `despawn_sphere` | bool | `true` | 是否应用 128m 自动消失球体 / Enable 128m despawn sphere |
| `exclusion_sphere` | bool | `true` | 是否应用 24m 安全球体 / Enable 24m exclusion sphere |
| `y_offset` | i32 | `0` | 相对玩家 Y 偏移，影响两个球体的半径投影 / Player-relative Y offset |
| `chunk_weight` | u32 | `0` | 判断区块属于遮罩所需的最少方块数 / Minimum block weight to mark a chunk inside |

**块遮罩 / Block mask**  
以玩家位置为中心（17×17 个区块，半径 `R_CHUNK = 8`），结合 despawn / exclusion 球体限制，得到需要统计的所有方块。  
Block mask covers a 17×17 chunk area centered on the player and removes blocks outside the despawn sphere or inside the exclusion sphere.

**区块遮罩 / Chunk mask**  
由于一个区块可能只有部分方块在 block mask 内，需要依赖 `chunk_weight`（0–255）来判定该区块是否算作“完全在遮罩中”。  
The chunk mask counts a chunk if the number of inside blocks exceeds `chunk_weight`.

图例仍可参考 `documentation/resources` 下的图片：`despawn-sphere.png`、`exclusion-sphere.png`、`block-mask.png`、`chunk-weight=*.png`。

### `[search]`

| 字段 Field | 类型 Type | 默认 Default | 说明 Description |
| --- | --- | --- | --- |
| `center_pos` | string | `"0c0,0c0"` | 搜索中心，可写成 `x,z`（方块）或 `Xcx,Zcz`（区块 + 相对位置） |
| `min_width` | u32 | `0` | 跳过中心的正方形宽度 / inner square width to skip |
| `max_width` | u32 | `1` | 搜索外部正方形宽度 / outer square width |
| `fine_search` | bool | `false` | 是否遍历每个 chunk 内的 256 个方块 / iterate all in-chunk positions |
| `min_block_size` | u32 | `0` | 最小块计数 / minimum block count |
| `max_block_size` | u32 | `73984` | 最大块计数（默认为全部块遮罩面积） |
| `min_chunk_size` | u32 | `0` | 最小区块数量 |
| `max_chunk_size` | u32 | `289` | 最大区块数量（默认为 17×17） |
| `output_file` | string | `"results.csv"` | 输出 CSV（可包含相对/绝对路径） |
| `append` | bool | `false` | 是否追加写入 / append instead of overwrite |

**搜索范围 / Search Area**  
程序以 `center_pos` 为起点，按照螺旋路径遍历 `max_width² - min_width²` 个区块。  
With fine search disabled，只会在每个区块内评估一个方块（即 `center_pos` 的 within-chunk 位置）；启用 fine search 则对所有 256 个 within-chunk 坐标进行评估。

**匹配条件 / Match Criteria**  
当某个位置的 `block_size` 处于 `[min_block_size, max_block_size]` 或 `chunk_size` 处于 `[min_chunk_size, max_chunk_size]` 时，就会被视为匹配并写入文件。  
Only one of the ranges needs to match for a position to be recorded.

## 输出文件 / Output CSV

默认 CSV 结构如下：

```
block-position;chunk-position;blockSize;chunkSize
0,0;0c0,0c0;4005/49640;18/222
```

- `block-position`：方块坐标（x,z）  
- `chunk-position`：`Xcx,Zcz` 格式  
- `blockSize`：`匹配块数/总块面积`（例如 `4005/49640`）  
- `chunkSize`：`匹配区块数/总区块面积`

若 `append = true`，数据会追加到文件末尾；否则在写入前会清空文件。程序在写入时自动补充表头。

## 示例配置 / Example Config

`slimefinder.toml`（位于项目根目录）提供了可直接运行的示例。你可以复制该文件并根据需要修改种子、搜索范围或输出路径。例如：

```toml
[mask]
world_seed = 123456789
despawn_sphere = true
exclusion_sphere = true
y_offset = 0
chunk_weight = 50

[search]
center_pos = "100c0,-200c0"
min_width = 0
max_width = 200
fine_search = false
min_block_size = 40000
max_block_size = 73984
min_chunk_size = 150
max_chunk_size = 289
output_file = "outputs/high-density.csv"
append = false
```

## 性能提示 / Performance Notes

- Rayon 会自动利用可用 CPU 核心。  
  Rayon automatically saturates available CPU cores.
- `fine_search = true` 会将任务数量乘以 256，适合对小范围进行深度扫描。  
  Use fine search only when refining promising regions.
- 设置较大的 `max_width` 时，建议确保搜索机子的内存和 CPU 充足。  
  Large max widths produce millions of jobs; be mindful of available resources.

祝你找到理想的史莱姆区块组合！  
Happy slime hunting!
