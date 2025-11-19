# Slimefinder (Rust + Rayon)

全新的 Rust 版本使用 Rayon 并行搜索 Minecraft 世界中的史莱姆区块组合，并通过 TOML 配置文件描述搜索参数。\
The Rust rewrite searches slime chunk clusters in parallel via Rayon and accepts settings from a TOML file.

![](documentation/resources/introduction.png)

## 快速开始 / Quick Start

```bash
cd Slimefinder
cargo run --release -- --config slimefinder.toml
```

- 首次没有配置文件时，程序会自动生成 `slimefinder.toml` 并退出，便于修改。  
  The tool writes a default `slimefinder.toml` on first launch and exits so you can edit it.
- 配置参数及注释为中英文双语，可根据需要调整种子、搜索范围和输出路径。  
  The config file uses bilingual comments describing the seed, search area, and output path.

## 配置格式 / Configuration

`[mask]` 定义世界种子、是否启用消失/排除球体、玩家高度偏移以及区块权重。  
`[search]` 定义搜索中心、内外方形宽度、是否遍历 256 个方块、块/区块数量阈值，以及 `results.csv` 输出位置。

示例配置存放在 `Slimefinder/slimefinder.toml`，与程序自动生成的默认值一致。

## 构建与运行 / Build & Run

- `cargo run --release -- --config slimefinder.toml`：执行搜索。
- `cargo fmt` / `cargo clippy`：格式化或静态检查（可选）。
- 输出 CSV 含 `block-position;chunk-position;blockSize;chunkSize` 列，使用 `;` 作为分隔符。

## 目录 / Layout

- `Slimefinder/src/main.rs`：核心实现（Rayon 并行、TOML 配置、结果输出）。
- `Slimefinder/slimefinder.toml`：示例配置。
- `Slimefinder/documentation`：沿用旧版的文字说明，可参考背景资料。

欢迎继续改进或扩展其它功能（如图像输出等）！  
Contributions are welcome for additional features like image generation or further tooling.
