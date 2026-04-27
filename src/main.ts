import './style.css'
const w = 600;
const h = 240;
let timeout = 300;
let logging = true;

let gamma = 1;
let epsilon = 0;
let alpha = 0.1;

const ctx = initdom();

const ROWS = 4, COLS = 10, TILE = 60;

let state = { x: 0, y: 3 };
let qTable: Record<string, Record<string, number> >;
resetQTable();

let totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
let resetting = false;
let history: Array<any> = []; // For Monte Carlo

function action(s: string): string {
  if (Math.random() < epsilon) {
    const actions = Object.keys(qTable[s]);
    return actions[Math.floor(Math.random() * actions.length)];
  }
  else {
    return highestAction(s);
  }
}

function highestAction(s: string): string {
  let ss = scores(s);
  let i = ss.indexOf(highestScore(s));  // 0, 1, 2 or 3
  return actions(s)[i];  // 'UP', 'DOWN', 'LEFT' or 'RIGHT'
}

function highestScore(s: string): number {
  return Math.max(...scores(s));
}

function actions(s: string): Array<string> {
  return Object.keys(qTable[s]);
}

function scores(s: string): Array<number> {
  return Object.values(qTable[s]);
}

function nextState(s: {x: number, y: number}, a: string) {
  let next = { x: s.x, y: s.y };
  if (a === 'UP') next.y = Math.max(0, s.y - 1);
  if (a === 'DOWN') next.y = Math.min(ROWS - 1, s.y + 1);
  if (a === 'LEFT') next.x = Math.max(0, s.x - 1);
  if (a === 'RIGHT') next.x = Math.min(COLS - 1, s.x + 1);
  
  let reward = -1;
  let done = false;

  // The Cliff
  if (next.y === 3 && next.x > 0 && next.x < COLS - 1) {
    reward = -100;
    next = { x: 0, y: 3 }; // Back to start
  } else if (next.x === COLS - 1 && next.y === 3) {
    reward = 0;
    done = true;
  }
  return { next, reward, done };
}

async function loop() {
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  let s = `${state.x},${state.y}`;
  let a = action(s);
  
  while (true) {
    let { next, reward, done } = nextState(state, a);  // calc the next state
    let sNext = `${next.x},${next.y}`;
    let aNext = action(sNext);  // calc what the action would be from that next state
    let q = qTable[s][a];
    
    if (algo !== 'montecarlo') {
      let target = algo === 'qlearning' ?
            reward + gamma * highestScore(sNext) :
            reward + gamma * qTable[sNext][aNext];
      qTable[s][a] += alpha * (target - q);

      log(s, a, ' ', sNext, aNext, ' ', 'q=' + q, 'target=' + target, 'newQ=' + qTable[s][a]);
    }
    else {
      history.push({ s, a, r: reward });
    }

    state = next;
    s = sNext;
    a = aNext;
    steps++;
    totalSteps++;
    totalReward += reward;

    draw();
    (document.getElementById('stats') as HTMLDivElement).innerText =
      `Total Steps: ${totalSteps} | Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;

    if (resetting) {
      totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
      state = { x: 0, y: 3 };
      s = '0,3';
      a = action(s);
      resetQTable();
      resetting = false;
    }
    else if (done || steps > 500) {
      if (algo === 'montecarlo') {
        let G = 0;
        for (let i = history.length - 1; i >= 0; i--) {
          G = history[i].r + gamma * G;
          qTable[history[i].s][history[i].a] += 0.05 * (G - qTable[history[i].s][history[i].a]);
        }
        history = [];
      }
      episode++;
      steps = 0;
      totalReward = 0;
      state = { x: 0, y: 3 };
      s = `0,3`;
      a = action(s);
    }
    await new Promise(r => setTimeout(r, timeout));
  }
}

function draw() {
  ctx.clearRect(0, 0, w, h);
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let s = `${x},${y}`;
      ctx.strokeStyle = '#ccc';
      ctx.strokeRect(x * TILE, y * TILE, TILE, TILE);
      
      // Draw Cliff
      if (y === 3 && x > 0 && x < COLS - 1) {
        ctx.fillStyle = '#ffcccc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
      // Draw Goal
      if (y === 3 && x === COLS - 1) {
        ctx.fillStyle = '#ccffcc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }

      if (qTable[s]) {
        // Draw a policy arrow
        const max = highestScore(s);    // e.g. -1.6
        const bestA = highestAction(s);  // e.g. 2
        ctx.fillStyle = "rgba(0, 0, 0, 1)";
        ctx.font = "20px Arial";
        const arrow = ({ UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[bestA] as string);
        if (max !== 0) {
            ctx.fillText(arrow, x * TILE + 22, y * TILE + 35);
        }

        // Draw the four q-values
        if (y !== 3 || x === 0) {
          ctx.fillStyle = '#000000';
          const q = qTable[s];
          if ('UP' in q) {
            const ww = Math.min(-q['UP'] * 60, TILE - 4);
            ctx.fillRect(x * TILE + TILE/2 - ww/2, y * TILE + 3, ww, 2);
          }
          if ('DOWN' in q) {
            const ww = Math.min(-q['DOWN'] * 60, TILE - 4);
            ctx.fillRect(x * TILE + TILE/2 - ww/2, (y+1) * TILE - 4, ww, 2);
          }
          if ('LEFT' in q) {
            const hh = Math.min(-q['LEFT'] * 60, TILE - 4);
            ctx.fillRect(x * TILE + 3, y * TILE + TILE/2 - hh/2, 2, hh);
          }
          if ('RIGHT' in q) {
            const hh = Math.min(-q['RIGHT'] * 60, TILE - 4);
            ctx.fillRect((x+1) * TILE - 4, y * TILE + TILE/2 - hh/2, 2, hh);
          }
        }
      }
    }
  }
  // Draw Agent
  ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
  ctx.beginPath();
  ctx.arc(state.x * TILE + TILE/2, state.y * TILE + TILE/2, 15, 0, Math.PI * 2);
  ctx.fill();
}

function resetSim() {
  resetting = true;
}

function resetQTable() {
  qTable = {};
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let s = `${x},${y}`;
      qTable[s] = {};
      const q = qTable[s];
      if (x > 0) q['LEFT'] = 0;
      if (y > 0) q['UP'] = 0;
      if (x < COLS - 1) q['RIGHT'] = 0;
      if (y < ROWS - 1) q['DOWN'] = 0;
    }
  }
}

function initdom() {
  const app = (document.querySelector<HTMLDivElement>('#app') as HTMLDivElement);

  const heading = document.createElement('h2');
  heading.textContent = 'Cliff Walking Reinforcement Learning';

  const controls = document.createElement('div');
  controls.className = 'controls';

  controls.append('Algorithm: ');

  const select = document.createElement('select');
  select.id = 'algoSelect';

  const options = [
    { value: 'qlearning', label: 'Q-Learning (Off-policy)' },
    { value: 'sarsa', label: 'SARSA (On-policy)' },
    { value: 'montecarlo', label: 'Monte Carlo (Every-visit)' },
  ];

  for (const opt of options) {
    const optionEl = document.createElement('option');
    optionEl.value = opt.value;
    optionEl.textContent = opt.label;
    select.appendChild(optionEl);
  }

  const button = document.createElement('button');
  button.textContent = 'Reset & Run';
  button.addEventListener('click', () => {
    resetSim();
  });

  const fastButton = document.createElement('button');
  fastButton.textContent = 'Faster';
  fastButton.addEventListener('click', () => {
    timeout /= 2;
  });

  const slowButton = document.createElement('button');
  slowButton.textContent = 'Slower';
  slowButton.addEventListener('click', () => {
    timeout *= 2;
  });

  const loggingButton = document.createElement('button');
  loggingButton.textContent = 'Toggle logging';
  loggingButton.addEventListener('click', () => {
    logging = !logging;
  });

  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.id = 'stats';
  stats.textContent = 'Total Steps: 0 | Episode: 0 | Steps: 0 | Reward: 0';

  const small = document.createElement('small');
  small.textContent =
    'Blue = Agent | Red = Cliff | Green = Goal | Yellow arrows = Policy';

  controls.appendChild(select);
  controls.appendChild(button);
  controls.appendChild(fastButton);
  controls.appendChild(slowButton);
  controls.appendChild(loggingButton);
  controls.appendChild(stats);
  controls.appendChild(small);

  const canvas = (document.createElement('canvas') as HTMLCanvasElement);
  canvas.id = 'gridCanvas';
  canvas.width = w;
  canvas.height = h;

  app.appendChild(heading);
  app.appendChild(controls);
  app.appendChild(canvas);
  return (canvas.getContext('2d') as CanvasRenderingContext2D);
}

function log(...args: any[]): void {
  if (logging) {
    console.log(...args);
  }
}

draw();
await new Promise(r => setTimeout(r, timeout));
loop();
