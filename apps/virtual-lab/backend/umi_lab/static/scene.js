function avatarColors(seed) {
    const colors = [['#7f996a', '#b1c493', '#d7e2bb', '#536c4b'], ['#867398', '#b4a2c4', '#e1d7eb', '#5f4e73'], ['#6a8e99', '#9dbdc4', '#d1e4e8', '#476b78'], ['#b18e63', '#d0b58b', '#eee1bf', '#7d6447'], ['#a2797e', '#c9a4a4', '#ecd9d2', '#794e5c'], ['#758aab', '#a5b5d0', '#d8e1ed', '#4e6286']];
    return colors[parseInt(seed.slice(0, 8) || '0', 16) % colors.length];
}
export function drawAvatar(canvas, seed) {
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const [dark, body, face, eye] = avatarColors(seed);
    ctx.fillStyle = '#ecede0';
    ctx.fillRect(0, 0, 44, 44);
    ctx.fillStyle = dark;
    ctx.fillRect(10, 12, 25, 24);
    ctx.fillRect(21, 5, 3, 9);
    ctx.fillStyle = body;
    ctx.fillRect(8, 10, 25, 22);
    ctx.fillStyle = face;
    ctx.fillRect(12, 14, 17, 12);
    ctx.fillStyle = eye;
    ctx.fillRect(15, 18, 3, 4);
    ctx.fillRect(24, 18, 3, 4);
    if (parseInt(seed.slice(-2) || '0', 16) % 2) {
        ctx.fillStyle = dark;
        ctx.fillRect(4, 17, 4, 8);
        ctx.fillRect(33, 17, 4, 8);
    }
    ctx.fillStyle = body;
    ctx.fillRect(15, 33, 14, 10);
    const emblem = parseInt(seed.slice(8, 16) || '0', 16);
    for (let bit = 0; bit < 9; bit++) {
        ctx.fillStyle = emblem & (1 << bit) ? eye : face;
        ctx.fillRect(19 + bit % 3 * 2, 35 + Math.floor(bit / 3) * 2, 2, 2);
    }
}
export function drawScene(canvas, state) {
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const box = (x, y, w, h, color) => { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); };
    const line = (points, color, width = 2) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.stroke();
    };
    const label = (value, x, y, color = '#8c9c7d', size = 8) => {
        ctx.fillStyle = color;
        ctx.font = `${size}px monospace`;
        ctx.fillText(value, x, y);
    };
    box(0, 0, 800, 360, '#edf0e4');
    box(0, 0, 800, 108, '#e6ecdc');
    box(0, 103, 800, 6, '#ccd6be');
    box(0, 109, 800, 3, '#f6f8ed');
    // Quiet floor tiles, rendered once while idle.
    for (let x = -360; x < 850; x += 42)
        line([[x, 112], [x + 360, 360]], '#e0e6d6', 1);
    for (let x = 0; x < 1200; x += 42)
        line([[x, 112], [x - 360, 360]], '#e0e6d6', 1);
    // Window and wall shelving.
    box(291, 21, 152, 67, '#c6d1b9');
    box(296, 26, 142, 55, '#f9faee');
    box(300, 30, 64, 47, '#d5e1d3');
    box(370, 30, 64, 47, '#d9e4d9');
    box(300, 61, 64, 16, '#c4d4bf');
    box(370, 57, 64, 20, '#c5d5c0');
    box(312, 48, 13, 13, '#c7d9c8');
    box(390, 43, 19, 14, '#ccddce');
    box(287, 84, 160, 5, '#b9c8aa');
    box(86, 56, 112, 5, '#acbc9d');
    box(90, 61, 4, 9, '#bbc8af');
    box(187, 61, 4, 9, '#bbc8af');
    [['#a7b493', 99, 27, 10, 29], ['#b8a8c5', 112, 34, 12, 22], ['#cebf9d', 127, 23, 9, 33], ['#99ad92', 140, 30, 13, 26]].forEach(v => box(v[1], v[2], v[3], v[4], v[0]));
    box(173, 38, 12, 18, '#a6b995');
    box(170, 34, 18, 5, '#8ca37a');
    box(546, 25, 90, 51, '#c8d3bc');
    box(550, 29, 82, 43, '#f5f6e9');
    label('OBSERVE', 568, 46, '#83956f', 9);
    label('INTERVENE', 560, 61, '#a6af96', 8);
    // Storage cabinet and a plant.
    box(68, 143, 73, 63, '#d9dfcd');
    box(63, 135, 77, 62, '#b4c1a5');
    box(67, 139, 69, 19, '#d3dcc5');
    box(67, 162, 69, 27, '#c8d2ba');
    box(97, 147, 10, 3, '#97a88a');
    box(97, 173, 10, 3, '#97a88a');
    box(70, 197, 5, 9, '#9eae90');
    box(130, 197, 5, 9, '#9eae90');
    box(83, 117, 22, 17, '#b4a486');
    box(79, 113, 30, 5, '#c8ba9b');
    box(91, 84, 4, 29, '#88a174');
    box(80, 91, 13, 9, '#92ad7e');
    box(93, 80, 14, 10, '#9bb787');
    box(77, 83, 10, 7, '#b1c39b');
    // Main experiment station: shadow, legs, front and tabletop.
    box(239, 236, 331, 17, '#d4ddc9');
    box(267, 195, 9, 49, '#a8b995');
    box(535, 195, 9, 49, '#a8b995');
    box(253, 172, 306, 40, '#bdccac');
    box(253, 166, 306, 10, '#d2ddc2');
    box(258, 176, 296, 26, '#c9d5bb');
    box(253, 162, 306, 6, '#e9eddc');
    box(257, 202, 297, 4, '#adbf9b');
    // Stimulus monitor.
    box(287, 111, 69, 51, '#a7b79a');
    box(291, 108, 61, 45, '#c3cfb6');
    box(295, 112, 53, 35, '#526c58');
    box(300, 117, 43, 24, '#657e65');
    label(state.running ? '7 > 3 ?' : 'READY_', 305, 131, '#e2ecd0', 8);
    box(314, 153, 11, 7, '#99ae8a');
    box(306, 159, 29, 3, '#b4c5a2');
    box(291, 170, 61, 9, '#e4e8d5');
    for (let x = 295; x < 348; x += 6)
        box(x, 172, 3, 2, '#b3c0a3');
    // Notebook and mug.
    box(477, 155, 34, 18, '#9cac8c');
    box(475, 152, 34, 18, '#f2eedb');
    line([[492, 153], [492, 169]], '#c5cdb6', 1);
    for (let y = 156; y < 167; y += 4)
        line([[478, y], [489, y]], '#cbd2bd', 1);
    box(527, 143, 12, 18, '#f7f5e6');
    box(539, 147, 4, 10, '#bbc9aa');
    box(528, 141, 10, 3, '#c6baa0');
    // Subject chair.
    box(391, 244, 64, 14, '#d3dcc5');
    box(401, 230, 43, 13, '#94ab7e');
    box(405, 241, 5, 23, '#a4b593');
    box(438, 241, 5, 23, '#a4b593');
    // Subject is an engineered model, represented by a small pixel robot.
    const blink = state.running && state.frame % 8 === 3;
    const bob = state.running ? state.frame % 2 * 2 : 0;
    const robot = (x, y) => {
        const [dark, body, face, eye] = avatarColors(state.seed ?? '0');
        box(x + 5, y + 55, 41, 7, '#c8d4bb');
        box(x + 10, y + 34, 28, 21, dark);
        box(x + 14, y + 39, 20, 11, face);
        box(x + 5, y + 36, 6, 17, body);
        box(x + 38, y + 36, 6, 17, body);
        box(x + 12, y + 54, 8, 7, dark);
        box(x + 30, y + 54, 8, 7, dark);
        box(x + 3, y + 8, 43, 29, dark);
        box(x + 7, y + 5, 35, 29, body);
        box(x + 10, y + 10, 29, 19, face);
        box(x + 15, y + 16, 4, blink ? 1 : 6, eye);
        box(x + 29, y + 16, 4, blink ? 1 : 6, eye);
        box(x + 22, y + 2, 3, 6, dark);
        box(x + 20, y - 2, 7, 5, '#c4b486');
        if (parseInt((state.seed ?? '0').slice(-2), 16) % 2) {
            box(x - 1, y + 14, 4, 13, dark);
            box(x + 46, y + 14, 4, 13, dark);
        }
        const emblem = parseInt((state.seed ?? '0').slice(8, 16) || '0', 16);
        for (let bit = 0; bit < 9; bit++)
            box(x + 21 + bit % 3 * 2, y + 41 + Math.floor(bit / 3) * 2, 2, 2, emblem & (1 << bit) ? eye : face);
    };
    if (state.ready)
        robot(397, 180 + bob);
    label('PARTICIPANT', 384, 287, '#8c9e7c', 8);
    // Tool carts appear when attached; the scene mirrors configuration.
    if (state.activity) {
        line([[464, 199], [500, 199], [500, 238], [604, 238]], '#a89bb9', 2);
        box(599, 205, 78, 62, '#d6d0dd');
        box(595, 201, 78, 57, '#b1a3c3');
        box(601, 207, 66, 34, '#e5dfed');
        box(607, 212, 54, 23, '#8e7fa5');
        const offset = state.running ? state.frame % 3 : 0;
        line([[610, 225], [619, 225], [623, 216 + offset], [629, 231], [634, 221], [641, 224], [657, 224]], '#ddd4ea', 2);
        box(608, 247, 5, 5, '#eee9f2');
        box(619, 247, 5, 5, '#eee9f2');
        box(650, 246, 11, 5, '#d9cfe5');
        box(604, 262, 6, 6, '#9f91b0');
        box(659, 262, 6, 6, '#9f91b0');
        label('ACTIVITY', 605, 285, '#9d8bab', 8);
    }
    if (state.behavior) {
        box(165, 233, 57, 43, '#cad6bb');
        box(162, 229, 57, 42, '#dce5cb');
        box(169, 235, 43, 25, '#f3f3df');
        for (let y = 241; y < 258; y += 6) {
            box(175, y, 4, 3, '#9eb586');
            box(184, y, 22, 2, '#cad7b7');
        }
        box(178, 225, 24, 7, '#9bad89');
        label('BEHAVIOR', 160, 290, '#8c9f79', 8);
    }
    if (state.ablation) {
        const active = state.running && (state.condition === 'ablation' || state.condition === 'random_control');
        box(583, 134, 43, 38, '#c3ae88');
        box(579, 130, 43, 37, '#decea9');
        box(585, 136, 31, 16, '#f0e5c8');
        box(592, 140, 7, 7, active ? '#bf855b' : '#b3b18d');
        box(605, 140, 6, 7, '#d0bf9c');
        box(587, 157, 25, 4, '#bda67f');
        line([[579, 154], [565, 154], [565, 119], [424, 119], [424, 176]], active ? '#b18a5f' : '#cec2a2', 2);
        box(416, 176, 16, 5, '#c7b18a');
        label(active ? 'ABLATING' : 'ABLATION', 576, 188, '#ad946e', 8);
    }
    // A wall clock and ventilation finish the room.
    box(709, 41, 23, 23, '#becbb0');
    box(712, 44, 17, 17, '#f3f5e6');
    line([[720, 47], [720, 53], [725, 53]], '#9eae8e', 2);
    for (let y = 76; y < 90; y += 4)
        box(702, y, 35, 2, '#c9d4be');
}
