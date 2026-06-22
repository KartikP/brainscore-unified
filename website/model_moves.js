// Recorded Qwen2.5-VL-7B (chain-of-thought) move sequences, captured on EC2 by
// scripts/vlm_game/record_moves.py at the validated config (GridGameEnv size=5,
// max_steps=20, the same setting under which the 7B scored 0.53 success).
//
// Each entry's (player, goal) is captured at play_game's own env.reset(), so the
// board here is EXACTLY the board the model navigated; game.js seeds its playable
// boards from these positions so "play it yourself" and "watch the model play"
// are the identical board. action index -> move: 0=up 1=down 2=left 3=right.
//
// Action traces verified to reconcile: e.g. seed 501 [3,2] --up--> [2,2] --right-->
// [2,3] --right--> [2,4]=goal (3 moves, optimal 3).
window.MODEL_MOVES = {
  "503": { "player": [0, 0], "goal": [2, 2], "actions": [1, 1, 3, 3],        "solved": true, "steps": 4, "optimal": 4 },
  "504": { "player": [1, 2], "goal": [2, 4], "actions": [3, 1, 3],           "solved": true, "steps": 3, "optimal": 3 },
  "505": { "player": [2, 2], "goal": [1, 4], "actions": [3, 0, 3],           "solved": true, "steps": 3, "optimal": 3 },
  "509": { "player": [1, 1], "goal": [2, 3], "actions": [1, 1, 3, 0, 3],     "solved": true, "steps": 5, "optimal": 3 },
  "511": { "player": [0, 1], "goal": [2, 3], "actions": [1, 1, 1, 3, 0, 3],  "solved": true, "steps": 6, "optimal": 4 }
};
