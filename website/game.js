// Playable version of the exact grid game the VLM played. Boards are the real
// (player, goal) layouts the model faced (GridGameEnv size=6, seeds 11/500/501/502),
// so you navigate the identical game. Rendering matches the Python env.
(function () {
  const SIZE = 6, CELL = 64, PAD = 6;
  const BG = '#ebebeb', GRID = '#c8c8c8', PLAYER = '#285adc', GOAL = '#28be46';
  // [player[r,c], goal[r,c], label]
  const BOARDS = [
    { player: [3, 3], goal: [5, 2], label: 'the rollout shown above (seed 11)' },
    { player: [0, 0], goal: [0, 4], label: 'eval board #1 (seed 500)' },
    { player: [1, 3], goal: [2, 0], label: 'eval board #2 (seed 501)' },
    { player: [5, 3], goal: [1, 3], label: 'eval board #3 (seed 502)' },
  ];
  let bi = 0, player, goal, moves, optimal, solved;

  const canvas = document.getElementById('game-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const status = document.getElementById('game-status');

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
        `For reference: the oracle always plays optimally; Qwen-VL-3B scored 0.0 on boards like this, 7B 0.13.`);
    } else {
      setStatus(`${moves} moves · ${manhattan(player, goal)} to go (optimal was ${optimal}).`);
    }
  }

  const DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
  document.querySelectorAll('[data-dir]').forEach(btn =>
    btn.addEventListener('click', () => { const d = DIRS[btn.dataset.dir]; move(d[0], d[1]); }));
  const nb = document.getElementById('game-new');
  if (nb) nb.addEventListener('click', () => { bi = (bi + 1) % BOARDS.length; load(bi); });

  // arrow keys (only when the canvas region is in view / focused-ish)
  window.addEventListener('keydown', (e) => {
    const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
    if (map[e.key]) {
      const rect = canvas.getBoundingClientRect();
      if (rect.top < window.innerHeight && rect.bottom > 0) { e.preventDefault(); const d = DIRS[map[e.key]]; move(d[0], d[1]); }
    }
  });

  load(0);
})();
