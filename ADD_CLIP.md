# 往主页添加一段 in-the-wild 结果

主页的 in-the-wild 片段列表来自 `data/clips.json`，不需要改 HTML。

## 1. 导出（在跑过 GravHMR 的机器上）

需要：GravHMR 的结果（FATC 的 `opt_results.pt` 或 core 的 `hmr4d_results.pt`）、GVHMR 预测 `hmr4d_results.pt`、ViPE 相机 `<seq>.npz`、管线用的那段输入视频，以及一个 GVHMR checkout（读人体模型）。

```bash
python tools/add_clip.py --gvhmr_root /path/to/GVHMR \
    --result outputs/run/fatc/huayun5/opt/huayun5/opt_results.pt \
    --hmr    outputs/run/hmr/huayun5/hmr4d_results.pt \
    --vipe   outputs/run/vipe/huayun5.npz \
    --video  inputs/huayun5.mp4 \
    --id huayun5 --sport Dancing
```

- 视频帧数必须和结果一一对应（就是管线跑的那段）。结尾跟踪丢了可以加 `--max_frames N` 只保留前 N 帧。
- 加 `--first` 让主页打开时默认显示这段。
- 世界坐标会自动按重力转正（ViPE 世界是 +y 朝下），不用手动处理。
- 删除一段：`python tools/add_clip.py --remove huayun5`

## 2. 发布

```bash
bash tools/publish.sh "add huayun5"
```

会重建 `docs/` 并 push，1–2 分钟后 https://jhl25-pixel.github.io/gravhmr/ 更新。推送需要仓库写权限（让仓库管理员在 Settings → Collaborators 里加你）。

## 注意

- 只放 GravHMR 跑得正常的片段；没有 GT，只是定性展示。
- 视频来源要能公开展示（自己拍的，或许可允许的素材）。
- 单段建议不超过 30 秒，`data/` 里每段约 1–3 MB。
