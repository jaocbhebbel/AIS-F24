---

# Reversi AI Game

## 项目简介
Reversi（黑白棋）是一款经典的棋盘策略游戏。本项目实现了一个基于 **强化学习策略网络** 的黑白棋人工智能（AI）。用户可以选择与 AI 对战，也可以让 AI 自行训练策略。提供了一个图形化用户界面（GUI）供玩家直观地与 AI 对战。

---

## 功能特性
- **AI 自训练**：
  - 使用强化学习策略（Q-learning）训练模型。
  - 支持模型的保存和加载。
- **人与 AI 对战**：
  - 提供图形化用户界面（GUI）。
  - 高亮玩家的合法落子位置。
  - 支持动态更新棋盘。
- **易于扩展**：
  - 使用 PyTorch 实现策略网络，可灵活调整模型架构或训练策略。

---

## 项目结构

```
reversi_ai_project/
│
├── game.py              # 黑白棋游戏逻辑
├── train.py             # 训练 AI 模型
├── play.py              # 图形化用户界面 (GUI)
├── model.py             # 策略神经网络 (Q-learning)
├── replay_buffer.py     # 回放缓冲区 (Replay Buffer)
├── main.py              # 入口文件，选择训练或对战
├── reversi_model.pth    # 保存的模型（训练完成后生成）
└── README.md            # 说明文档
```

---

## 安装说明

### 环境依赖

1. 安装 Python 3.7 或更高版本。
2. 安装依赖库：
   ```bash
   pip install torch tkinter numpy
   ```

---

## 使用方法

### 1. 运行项目

在终端中运行主程序：
```bash
python main.py
```

### 2. 操作说明

#### **主菜单**
程序启动后，您可以选择：
1. `Train AI`：让 AI 自行训练，生成策略模型。
2. `Play against AI`：进入 GUI 与 AI 对战。

#### **AI 训练**
选择 `1` 后，程序会开始训练 AI：
- 每隔 100 局显示一次训练损失。
- 训练完成后，模型会保存为 `reversi_model.pth`。

#### **人与 AI 对战**
选择 `2` 后，会启动图形化界面：
- 白棋（O）是玩家，黑棋（X）是 AI。
- 点击棋盘合法位置完成落子。
- 非法位置点击无响应。
- 棋盘会高亮显示玩家的合法位置。

---

## 游戏规则
1. 棋盘大小为 8x8。
2. 双方轮流落子：
   - 白棋先手（玩家），黑棋后手（AI）。
   - 每步必须翻转至少一个对方棋子，否则无法落子。
3. 棋局结束：
   - 当双方都无合法步时结束。
   - 以棋子数量多的一方获胜。

---

## 注意事项

1. 如果训练损失值过高或模型效果不佳，可以调整以下内容：
   - 增加训练迭代次数。
   - 调整奖励函数的范围。
2. 如果模型文件丢失，需重新训练模型。

---

## 未来改进方向

- 增加玩家与玩家对战模式。
- 实现模型评估工具，用于分析训练效果。
- 提升 GUI 的美观性，例如添加棋盘背景和动态效果。

---

## 许可证
本项目基于 MIT 许可证开源，欢迎自由使用和修改！

---

如有问题或建议，请随时联系或提交 issue！