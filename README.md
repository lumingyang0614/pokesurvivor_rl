# PokeSurvivor RL

用強化學習 (PPO) 訓練 AI 玩 [PokéSurvivor](https://pokesurvivor.com/en/)。

## 運作方式

```
Python (stable-baselines3 PPO)
   │  action: 0=站著, 1-8=八方向移動
   ▼
Playwright ── Chromium ── pokesurvivor.com (原版遊戲)
                  ▲
          bridge.js（注入頁面）
```

- **不靠截圖辨識**：`pokesurvivor_rl/bridge.js` 在遊戲載入前注入，直接讀取遊戲的全域 `G` 物件
  （玩家、敵人、敵方子彈、掉落物、地形、危險區）。
- **時間由 Python 控制**：接管 `requestAnimationFrame`，每一步固定推進 3 個 1/30 秒的畫格，
  所以訓練可以比真實時間快很多倍（單一瀏覽器約 20 倍速），而觀看時再放慢回真實速度。
- **AI 負責移動**：遊戲本身會自動攻擊。升級卡片、招式連結 (Bond)、道具交換由簡單規則自動選擇。
- 以「訪客」身份遊玩，並停用排行榜上傳，訓練不會灌爆官方排行榜。

### 觀測 (174 維 × 4 幀堆疊)
- 16 個方向扇區 × 8 個通道：最近敵人、敵人密度、朝自己飛來的子彈、經驗/金幣、特殊道具、牆壁距離、危險區、Boss
- 14 個玩家特徵：血量、等級、速度、經驗、時間、敵人數…
- 最近 8 隻敵人的相對位置、傷害、剩餘血量

### 獎勵
存活每秒 +0.6、損血（整條血 -6）、擊殺 +0.03、收集經驗（每升一級共 +1）、擊敗 Boss +3、死亡 -5

## 使用

```bash
pip install -r requirements.txt
python -m playwright install chromium

# 訓練（12 個平行瀏覽器，背景無頭模式）
python train.py --envs 12 --steps 4000000
tensorboard --logdir runs          # 看 game/survive_sec 曲線

# 讓 AI 玩給你看（開瀏覽器視窗，真實速度）
python play.py
python play.py --record videos     # 同時錄成 .webm
python play.py --random            # 對照組：亂走
python play.py --speed 0           # 不限速快速跑
```

### 永久強化（遊戲裡的「Mejoras／強化」選單）
每局結束時遊戲會依存活時間、擊殺數發金幣。下一局開始前，AI 會自動把金幣花在該寶可夢的永久強化上
（攻擊、體力、冷卻、防禦、磁鐵、回復、爆擊、速度、經驗、金幣加成），優先買「便宜又想要」的項目。
金幣與強化等級存在 `progress/*.json`，關掉瀏覽器也不會消失：
- `progress/play.json`：觀看模式用（`--save` 可換檔、`--no-upgrade` 關閉自動強化）
- `progress/train_env*.json`：每個訓練環境各自累積

`--partner` 可換夥伴（圖鑑編號，例如 1 妙蛙種子、4 小火龍、7 傑尼龜）。
Checkpoints 存在 `checkpoints/<name>/`，`--resume` 可接續訓練。
