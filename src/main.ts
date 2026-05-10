// A reinforcement learning demo on the cliff walking grid world.
// You can see it in action here: https://dominicprior.github.io/rl/

// Still under construction...

// Here are some future enhancements.
//
// - Pause and single-step buttons.
// - A way to rewind the demo.
// - Earlier arrows.
// - Optional cliff.
// - Fix Monte Carlo.
// - Variable w and h.
// - Variable gamma, epsilon and alpha.
// - A graph of the episode lengths.
// - Split the animation into moving the agent and updating Q.
// - Add a delay for the cliff and the goal.

import './style.css'
const w = 600;
const h = 240;
let timeout = 100;
let logging = false;

// Gamma is the discount factor for future rewards.  A value of 1 mean rewards
// contribute equally to the overall return regardless of when they occur.
let gamma = 1;

// Epsilon is the probability of exploring randomly, as opposed to greedily
// choosing the action with the highest Q value.  An epsilon of zero means it
// always chooses the highest, instead of exploring, which means Q-Learning
// and Sarsa are the same.  In this demo, the
// initial Q values are high (by being zero rather than negative), which
// makes the agent think every path is amazing until it tries it and is "disappointed."
// This forces it to traverse almost every arc in the graph at least once
// before it settles on a favourite.
let epsilon = 0;

// The nudge factor for updating a Q value from a successor Q value.
let alpha = 0.1;

const ctx = initdom();

const ROWS = 4, COLS = 10, TILE = 60;
let qValueScale = TILE;  // pixels per unit q-value
let maxNegQValue = 0; // apart from ones bordering the cliff

type dir = 'LEFT' | 'UP' | 'RIGHT' | 'DOWN';
type pair = [number, number];

let state = [3, 0] as pair;  // y first, i.e. row first
// The qTable is an array of rows.
// Each row is an array of mappings from direction to q-value.
// i.e. y first again.
let qTable: Array<Array<Record<dir, number> > >;  // e.g. qTable[0][0]['DOWN'] would be a Q value.
resetQTable();

let totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
let resetting = false;

// The q-values for state s.  The pair is the y then the x.
function qValues(s: pair): Record<dir, number> {
  return qTable[s[0]][s[1]];
}

// The epsilon-greedy action ('UP', 'DOWN', 'LEFT' or 'RIGHT')
// from the state, s (a string containing the x and y, e.g. '0,3').
function choose(s: pair): dir {
  if (Math.random() < epsilon) {
    const aa = actions(s);
    return aa[Math.floor(Math.random() * aa.length)];
  }
  else {
    return highestAction(s);
  }
}

// The action with the highest Q value.
// (In the event of a tie choose the first action).
function highestAction(s: pair): dir {
  let ss = scores(s);
  let i = ss.indexOf(highestScore(s));  // 0, 1, 2 or 3
  return actions(s)[i];  // 'UP', 'DOWN', 'LEFT' or 'RIGHT'
}

// The highest Q value from the state, s.
function highestScore(s: pair): number {
  return Math.max(...scores(s));
}

// The possible actions from the state, s.
function actions(s: pair): Array<dir> {
  return Object.keys(qValues(s)) as Array<dir>;
}

// The Q values from the state, s.
function scores(s: pair): Array<number> {
  return Object.values(qValues(s));
}

// The result of taking action a from state s.
function nextState(s: pair, a: dir) {
  let y = s[0];
  let x = s[1];
  if (a === 'UP')    y--;
  if (a === 'DOWN')  y++;
  if (a === 'LEFT')  x--;
  if (a === 'RIGHT') x++;
  
  let reward = -1;
  let done = false;

  // The Cliff
  if (y === 3 && x > 0 && x < COLS - 1) {
    reward = -100;
    y = 3; // Back to start
    x = 0;
  }
  else if (x === COLS - 1 && y === 3) {
    reward = 0;
    done = true;
  }
  const next = [y, x] as pair;
  return { next, reward, done };
}

function updateQ(s: pair, a: dir, algo: string, next: pair, aNext: dir, reward: number): void {
  const q = algo === 'qlearning' ? highestScore(next) : qValues(next)[aNext];
  let target = reward + gamma * q;
  qValues(s)[a] += alpha * (target - qValues(s)[a]);
  maybe_decrease_qValueScale(qValues(s)[a], reward);
  log(s, a, ' ', next, aNext, ' ', 'q=' + q, 'target=' + target, 'newQ=' + qValues(s)[a]);
  log(maxNegQValue);
}

function step(s: pair, a: dir, algo: string): [pair, dir, number, boolean] {
  let { next, reward, done } = nextState(state, a);  // Calc the next state...
  let aNext = choose(next);  // ...and choose what the action would be from that next state.
  updateQ(s, a, algo, next, aNext, reward);
  return [next, aNext, reward, done];
}

async function loop() {
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  let a = choose(state);

  while (true) {
    let reward: number;
    let done; Boolean;
    [state, a, reward, done] = step(state, a, algo);
    steps++;
    totalSteps++;
    totalReward += reward;

    draw();
    (document.getElementById('stats') as HTMLDivElement).innerText =
      `Total Steps: ${totalSteps} | Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;

    if (resetting) {
      totalSteps = 0, episode = 0, steps = 0, totalReward = 0, maxNegQValue = 0;
      state = [3, 0];  // row, col
      a = choose(state);
      resetQTable();
      resetting = false;
    }
    else if (done || steps > 5000) {
      episode++;
      steps = 0;
      totalReward = 0;
      state = [3, 0];
      a = choose(state);
    }
    await new Promise(r => setTimeout(r, timeout));
  }
}

// Make sure the q-values lines fit inside the tiles.
function maybe_decrease_qValueScale(q: number, reward: number): void {
  if (-q > maxNegQValue && reward !== -100) {
    maxNegQValue = -q;
    if (maxNegQValue * qValueScale > 0.9 * TILE) {
      qValueScale /= 2;
    }
  }
}

function draw() {
  ctx.clearRect(0, 0, w, h);
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let s = [y, x] as pair;
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

      drawArrow(ctx, s);

      // Draw the four q-values
      if (y !== 3 || x === 0) {
        ctx.fillStyle = '#000000';
        const q = qValues(s);
        if ('UP' in q) {
          const ww = Math.min(-q['UP'] * qValueScale, TILE - 4);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, y * TILE + 3, ww, 2);
        }
        if ('DOWN' in q) {
          const ww = Math.min(-q['DOWN'] * qValueScale, TILE - 4);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, (y+1) * TILE - 4, ww, 2);
        }
        if ('LEFT' in q) {
          const hh = Math.min(-q['LEFT'] * qValueScale, TILE - 4);
          ctx.fillRect(x * TILE + 3, y * TILE + TILE/2 - hh/2, 2, hh);
        }
        if ('RIGHT' in q) {
          const hh = Math.min(-q['RIGHT'] * qValueScale, TILE - 4);
          ctx.fillRect((x+1) * TILE - 4, y * TILE + TILE/2 - hh/2, 2, hh);
        }
      }
    }
  }
  // Draw Agent
  ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
  ctx.beginPath();
  ctx.arc(state[1] * TILE + TILE/2, state[0] * TILE + TILE/2, 15, 0, Math.PI * 2);
  ctx.fill();
}

// Draw an arrow unless the top two actions still have zero q values.
function drawArrow(ctx: CanvasRenderingContext2D, s: pair) {
  const y = s[0], x = s[1];
  const q = qValues(s);  // e.g. { DOWN: 3, RIGHT: 4 }
  // Find the indexes of the actions in the reverse order of their Q values.
  const aa: Array<dir> = actions(s).toSorted(  // e.g. ['RIGHT', 'DOWN']
    (a,b) => q[b] - q[a]
  );

  const bestA = aa[0];  // e.g. 'UP'
  ctx.fillStyle = "#f80";
  ctx.font = "20px Arial";
  const str = arrowStr(bestA);
  if (q[aa[1]] !== 0) {
      ctx.fillText(str, x * TILE + 22, y * TILE + 35);
  }
}

function arrowStr(dir: dir): string {
  return { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[dir];
}

function resetSim() {
  resetting = true;
}

function resetQTable() {
  qTable = [];
  for (let y = 0; y < ROWS; y++) {
    qTable.push([]);
    for (let x = 0; x < COLS; x++) {
      qTable[y].push({} as Record<dir, number>);
      const q = qTable[y][x];
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
    'Blue = Agent | Pink = Cliff | Green = Goal | Orange arrows = Policy | Black lines = Q values';

  const srctext = document.createElement('small');
  srctext.innerHTML =
    '<a href="https://github.com/dominicprior/rl/blob/main/src/main.ts">See on GitHub</a>';

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
  app.appendChild(srctext);
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
