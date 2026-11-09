# Third-party notices in the Side Car source tree

Keep the licence files and notices listed here when redistributing their corresponding sources. This list covers each direct child of `third_party/`; other transitive dependencies are governed by their own notices.

| Folder | Origin | Licence and local text |
| --- | --- | --- |
| `third_party/antirez-ds4` | [antirez/ds4](https://github.com/antirez/ds4), pinned in [NOTICE](third_party/antirez-ds4/NOTICE.md) | MIT: [LICENSE](third_party/antirez-ds4/LICENSE) |
| `third_party/ggml` | [ggml](https://github.com/ggml-org/ggml), ggml-common headers | MIT: [LICENSE](third_party/ggml/LICENSE) |
| `third_party/ggml-tq` | [agentionai/llama.cpp](https://github.com/agentionai/llama.cpp), pinned in [NOTICE](third_party/ggml-tq/NOTICE.md) | MIT: [LICENSE](third_party/ggml-tq/LICENSE) |
| `third_party/libstrix` | [asf0/libstrix](https://github.com/asf0/libstrix), source attribution in [NOTICE](third_party/libstrix/NOTICE.md) | MIT: [LICENSE](third_party/libstrix/LICENSE) |
| `third_party/project-maya` | [Project Maya](https://github.com/mw00/project-maya), pinned in [NOTICE](third_party/project-maya/NOTICE.md) | MIT: [LICENSE](third_party/project-maya/LICENSE) |
| `third_party/sfnx` | [sFNX](https://github.com/johnfink8/sfnx), pinned in [NOTICE](third_party/sfnx/NOTICE.md) | MIT: [LICENSE](third_party/sfnx/LICENSE) |

The Mac CUDA bridge under `sidecar-shim/` uses [tinygrad](https://github.com/tinygrad/tinygrad); its [MIT licence text](sidecar-shim/LICENSE.tinygrad) and [source-pinned notice](sidecar-shim/NOTICE.md) accompany the shim. The shim began as the `tgcu` CUDA-runtime shim by 0xSero, from [qwen38-flash-next-3090-usb-offload](https://github.com/sybil-solutions/qwen38-flash-next-3090-usb-offload) at commit `d134c011bbb98bbe69941ca907d82f998320c6e4`, under MIT: [LICENSE.0xSero](sidecar-shim/LICENSE.0xSero). The [`driver/`](driver/NOTICE.md) folder has its own notice. The root [LICENSE](LICENSE) covers this project's own source, not a replacement for third-party notices.
