// Playable version of the exact grid game the VLM played. Boards are the real
// (player, goal) layouts Qwen2.5-VL-7B faced (GridGameEnv size=5, eval seeds),
// seeded from the recorded move data in model_moves.js so "play it yourself" and
// "watch the model play" are the identical board. Rendering matches the Python env.
(function () {
  const SIZE = 5;
  const BG = '#ebebeb', GRID = '#c8c8c8', PLAYER = '#285adc', GOAL = '#28be46';
  // Boards mirror window.MODEL_MOVES (model_moves.js): seed -> [player, goal].
  // Mix of optimal solves (503/504/505) and imperfect-but-solved detours (509/511).
  const BOARDS = [
    { seed: 504, player: [1, 2], goal: [2, 4], label: 'eval board (seed 504): the 7B solved this optimally' },
    { seed: 503, player: [0, 0], goal: [2, 2], label: 'eval board (seed 503): corner to centre, solved optimally' },
    { seed: 505, player: [2, 2], goal: [1, 4], label: 'eval board (seed 505): solved optimally' },
    { seed: 509, player: [1, 1], goal: [2, 3], label: 'eval board (seed 509): the 7B solved it but took a detour (5 vs 3)' },
    { seed: 511, player: [0, 1], goal: [2, 3], label: 'eval board (seed 511): solved with a detour (6 vs 4)' },
  ];
  // action index -> [dRow, dCol], matching grid_game _DELTA {0:up,1:down,2:left,3:right}
  const DIRS_BY_INDEX = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  let bi = 0, player, goal, moves, optimal, solved;

  const canvas = document.getElementById('game-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const status = document.getElementById('game-status');
  const CELL = Math.floor(canvas.width / SIZE);   // derive cell size from canvas
  const PAD = Math.round(CELL * 0.09);

  function manhattan(a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]); }

  function load(i) {
    const b = BOARDS[i];
    player = b.player.slice(); goal = b.goal.slice();
    moves = 0; solved = false; optimal = manhattan(player, goal);
    draw();
    setStatus(`Board: ${b.label}. Get the blue player to the green goal. Optimal = ${optimal} moves. Use arrow keys or the buttons.`);
  }

  function fillCell(r, c, color) {
    ctx.fillStyle = color;
    ctx.fillRect(c * CELL + PAD, r * CELL + PAD, CELL - 2 * PAD, CELL - 2 * PAD);
  }

  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    for (let i = 0; i <= SIZE; i++) {
      ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, SIZE * CELL); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(SIZE * CELL, i * CELL); ctx.stroke();
    }
    fillCell(goal[0], goal[1], GOAL);
    fillCell(player[0], player[1], PLAYER);
  }

  function setStatus(t) { if (status) status.textContent = t; }

  function move(dr, dc) {
    if (solved) return;
    const nr = player[0] + dr, nc = player[1] + dc;
    if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) return;
    player = [nr, nc]; moves++;
    draw();
    if (player[0] === goal[0] && player[1] === goal[1]) {
      solved = true;
      const eff = (optimal / moves);
      setStatus(`Solved in ${moves} moves (optimal ${optimal}, efficiency ${eff.toFixed(2)}). ` +
        `For reference: the oracle always plays optimally; Qwen-VL-3B scored 0.0 on boards like this, the 7B with chain-of-thought 0.53.`);
    } else {
      setStatus(`${moves} moves · ${manhattan(player, goal)} to go (optimal was ${optimal}).`);
    }
  }

  const DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
  document.querySelectorAll('[data-dir]').forEach(btn =>
    btn.addEventListener('click', () => { stopReplay(); const d = DIRS[btn.dataset.dir]; move(d[0], d[1]); }));
  const nb = document.getElementById('game-new');
  if (nb) nb.addEventListener('click', () => { stopReplay(); bi = (bi + 1) % BOARDS.length; load(bi); });

  // ---- replay the 7B model's recorded moves ----
  let replayTimer = null;
  function stopReplay() { if (replayTimer) { clearInterval(replayTimer); replayTimer = null; } }
  function watchModel() {
    stopReplay();
    const b = BOARDS[bi];
    const rec = (window.MODEL_MOVES || {})[String(b.seed)];
    if (!rec || !rec.actions) {
      setStatus('Model replay for this board has not been recorded yet.');
      return;
    }
    load(bi);  // reset to the start
    let i = 0;
    setStatus(`Watching Qwen-VL-7B (chain-of-thought) play this board…`);
    replayTimer = setInterval(() => {
      if (i >= rec.actions.length || solved) {
        stopReplay();
        if (rec.solved) {
          const verdict = rec.steps === rec.optimal
            ? `optimally (${rec.steps} = optimal ${rec.optimal})`
            : `in ${rec.steps} moves (optimal ${rec.optimal}): a detour, but it corrected course and reached the goal`;
          setStatus(`Qwen-VL-7B solved it ${verdict}.`);
        } else {
          setStatus(`Qwen-VL-7B did NOT reach the goal (${rec.steps} moves). Watch where its spatial reasoning drifts.`);
        }
        return;
      }
      const d = DIRS_BY_INDEX[rec.actions[i]]; i++;
      if (d) move(d[0], d[1]);
    }, 650);
  }
  const wb = document.getElementById('game-watch');
  if (wb) wb.addEventListener('click', watchModel);

  // arrow keys (only when the canvas region is in view)
  window.addEventListener('keydown', (e) => {
    const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
    if (map[e.key]) {
      const rect = canvas.getBoundingClientRect();
      if (rect.top < window.innerHeight && rect.bottom > 0) { e.preventDefault(); stopReplay(); const d = DIRS[map[e.key]]; move(d[0], d[1]); }
    }
  });

  load(0);
})();
