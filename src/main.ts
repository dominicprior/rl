// A reinforcement learning demo on the cliff walking grid world.
// You can see it in action here: https://dominicprior.github.io/rl/

// Still under construction...

// Here are some future enhancements.
//
// - A graph of the episode lengths.
// - Split the animation into moving the agent and updating Q.
// - Add a delay for the cliff and the goal.

import './style.css'
type dir = 'LEFT' | 'UP' | 'RIGHT' | 'DOWN';
type pair = [number, number];  // for storing a state
type historyItem = [ pair, dir, number,     // state, action, Q value
          number, number, number, number ]; // stats
type param = [label: string, value: number, min: number, max: number];

const TILE = 60;  // tile size

let timeout = 100;
let logging = false;

let params: Record<string, number> = {};

let paramData: Record<string, param> = {

  // Gamma is the discount factor for future rewards.  A value of 1 mean rewards
  // contribute equally to the overall return regardless of when they occur.

  gamma: ['Gamma', 1,   0, 1],

  // Epsilon is the probability of exploring randomly, as opposed to greedily
  // choosing the action with the highest Q value.  An epsilon of zero means it
  // always chooses the highest, instead of exploring, which means Q-Learning
  // and Sarsa are the same.  In this demo, the
  // initial Q values are high (by being zero rather than negative), which
  // makes the agent think every path is amazing until it tries it and is "disappointed."
  // This forces it to traverse almost every arc in the graph at least once
  // before it settles on a favourite.

  epsilon: ['Epsilon', 0,   0, 1],

  // Alpha is the nudge factor for updating a Q value from a successor Q value.

  alpha: ['Alpha', 0.1,   0, 1],

  rows: ['Rows', 4,       1, 10],
  cols: ['Columns', 10,   1, 10],

  init_y: ['Init Y', 3,   0, 10],
  init_x: ['Init X', 0,   0, 10],

  cliff_start: ['Cliff Start', 1,   0, 10],
  cliff_end:   ['Cliff End',   8,   0, 10],

  goal_y: ['Goal Y', 3,   0, 10],
  goal_x: ['Goal X', 9,   0, 10],

  cliff_penalty: ['Cliff Penalty', 100,   0, 500],
};

for (const [id, param] of Object.entries(paramData)) {
  params[id] = param[1];
}

let qValueScale = TILE;  // pixels per unit q-value
let maxNegQValue = 0; // apart from ones bordering the cliff

// The qTable is an array of rows.
// Each row is an array of cells.
// Each cell is a mapping from direction to q-value.
// y first again.
let qTable: Array<Array<Record<dir, number> > >;  // e.g. qTable[0][0]['DOWN'] would be a Q value.

let state: pair;  // y first, i.e. row first
let action: dir;

let timeoutId: number | null = null;

let totalSteps = 0, episode = 0, steps = 0, totalReward = 0;

let history: Array<historyItem>;

let app: HTMLDivElement;
let boxes: HTMLDivElement;
let ctx: CanvasRenderingContext2D;
let canvas: HTMLCanvasElement;

function isCliff(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return y === params.rows - 1 && x >= params.cliff_start && x <= params.cliff_end;
}

function isGoal(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return y === params.goal_y && x === params.goal_x;
}

function reset_qTable_and_state(): void {
  history = [];
  totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
  resetQTable();
  state = [params.init_y, params.init_x];
  action = chooseAction(state);
}

// The q-values for state s.  The pair is the y then the x.
function qValues(s: pair): Record<dir, number> {
  return qTable[s[0]][s[1]];
}

// The epsilon-greedy action ('UP', 'DOWN', 'LEFT' or 'RIGHT')
// from the state, s (a string containing the x and y, e.g. '0,3').
function chooseAction(s: pair): dir {
  if (Math.random() < params.epsilon) {
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
function lookupNextState(s: pair, a: dir) {
  let y = s[0];
  let x = s[1];
  if (a === 'UP')    y--;
  if (a === 'DOWN')  y++;
  if (a === 'LEFT')  x--;
  if (a === 'RIGHT') x++;

  let reward = -1;
  let done = false;

  if (isCliff([y, x])) {
    reward = -params.cliff_penalty;
    y = params.init_y;
    x = params.init_x;
  }
  else if (isGoal([y, x])) {
    reward = 0;
    episode++;
    steps = 0;
    totalReward = 0;
    y = params.init_y;
    x = params.init_x;
    done = true;
  }
  const nextState = [y, x] as pair;
  return { nextState, reward, done };
}

function updateQ(s: pair, a: dir, next: pair, nextAction: dir, reward: number, done: boolean): void {
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  const q = done ? 0 :
              algo === 'qlearning' ? highestScore(next) : qValues(next)[nextAction];
  let targetValue = reward + params.gamma * q;
  qValues(s)[a] += params.alpha * (targetValue - qValues(s)[a]);
  maybe_decrease_qValueScale(qValues(s)[a], reward);
  log(s, a, ' ', next, nextAction, ' ', 'q=' + q, 'target=' + targetValue, 'newQ=' + qValues(s)[a]);
  log(maxNegQValue);
}

function step(): boolean {
  let { nextState, reward, done } = lookupNextState(state, action);
  let nextAction = chooseAction(nextState);
  history.push([[state[0], state[1]], action, qValues(state)[action],
                  totalSteps, episode, steps, totalReward]);
  totalReward += reward;
  totalSteps++;
  steps++;
  updateQ(state, action, nextState, nextAction, reward, done);
  state = nextState;
  action = nextAction;
  draw();
  return done;
}

function scheduleNext() {
  timeoutId = setTimeout(() => {
    const done = step();
    if (done) {
      timeoutId = null;
    }
    else {
      scheduleNext();
    }
  }, timeout);
}

function stop_and_reset() {
  stop_animation();
  reset_qTable_and_state();
  draw();
}

function stop_animation() {
  if (timeoutId !== null) {
    clearTimeout(timeoutId);
    timeoutId = null;
  }
}

function resume_animation() {
  if (timeoutId === null) scheduleNext();
}

function singleStep() {
  stop_animation();
  step();
}

function back() {
  stop_animation();
  if (history.length !== 0) {
    const h = history.pop() as historyItem;
    state  = h[0];
    action = h[1];
    [qValues(state)[action],
        totalSteps, episode, steps, totalReward ] = (h.slice(2) as Array<number>);
    draw();
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
  ctx.clearRect(0, 0, params.cols * TILE, params.rows * TILE);
  for (let y = 0; y < params.rows; y++) {
    for (let x = 0; x < params.cols; x++) {
      let s = [y, x] as pair;
      ctx.strokeStyle = '#ccc';
      ctx.strokeRect(x * TILE, y * TILE, TILE, TILE);

      if (isCliff(s)) {
        ctx.fillStyle = '#ffcccc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }

      if (isGoal(s)) {
        ctx.fillStyle = '#ccffcc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }

      if (x === params.init_x && y === params.init_y) {
        ctx.fillStyle = '#ffffbb';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
      drawArrow(ctx, s);

      // Draw the four q-values
      if (!isCliff(s) && !isGoal(s)) {
        ctx.fillStyle = '#ff0000';
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
  ctx.arc(state[1] * TILE + TILE/2, state[0] * TILE + TILE/2, TILE/4, 0, Math.PI * 2);
  ctx.fill();
  // Write stats
  (document.getElementById('stats') as HTMLDivElement).innerText =
      `Total Steps: ${totalSteps} | Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;
}

function drawArrow(ctx: CanvasRenderingContext2D, s: pair) {
  const y = s[0], x = s[1];
  const q = qValues(s);  // e.g. { DOWN: 3, RIGHT: 4 }
  // Find the indexes of the actions in the reverse order of their Q values.
  const aa: Array<dir> = actions(s).toSorted(  // e.g. ['RIGHT', 'DOWN']
    (a,b) => q[b] - q[a]
  );

  const bestA = aa[0];  // e.g. 'UP'
  ctx.font = `${TILE/3}px Arial`;
  // ctx.font = "20px Arial";
  const str = arrowStr(bestA);
  if (q[aa[1]] === 0) {
    ctx.fillStyle = "#ddd";  // if still a tie
  }
  else {
    ctx.fillStyle = "#f80";  // if a definite choice
  }
  const isThin = bestA === 'UP' || bestA === 'DOWN';
  ctx.fillText(str, (x + (isThin ? 0.42 : 0.35)) * TILE,
        (y + 0.59) * TILE);
}

function arrowStr(dir: dir): string {
  return { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[dir];
}

function resetQTable() {
  qTable = [];
  for (let y = 0; y < params.rows; y++) {
    qTable.push([]);
    for (let x = 0; x < params.cols; x++) {
      qTable[y].push({} as Record<dir, number>);
      const q = qTable[y][x];
      if (x > 0) q['LEFT'] = 0;
      if (y > 0) q['UP'] = 0;
      if (x < params.cols - 1) q['RIGHT'] = 0;
      if (y < params.rows - 1) q['DOWN'] = 0;
    }
  }
}

function initdom() {
  app = (document.querySelector<HTMLDivElement>('#app') as HTMLDivElement);

  const heading = document.createElement('h2');
  heading.textContent = 'Cliff Walking Reinforcement Learning';

  const controls = document.createElement('div');
  boxes = document.createElement('div');
  boxes.id = 'boxes';

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

  addButton(controls, 'Faster', () => { timeout /= 2 });
  addButton(controls, 'Slower', () => { timeout *= 2 });
  addButton(controls, 'Toggle logging', () => { logging = !logging; });
  addButton(controls, 'Pause', stop_animation);
  addButton(controls, 'Resume', resume_animation);
  addButton(controls, 'Step', singleStep);
  addButton(controls, 'Back', back);
  addButton(controls, 'Reset', stop_and_reset);
  addButton(controls, 'Tiny', () => {
    setParam('cols', 4);
    setParam('rows', 1);
    clampParams();
    setParam('cliff_end', 0);
    createCanvas();
    stop_and_reset();
  });
  addButton(controls, 'Small', () => {
    setParam('cols', 5);
    setParam('rows', 2);
    clampParams();
    setParam('cliff_end', 3);
    createCanvas();
    stop_and_reset();
  });
  addButton(controls, 'Medium', () => {
    setParam('cols', 6);
    setParam('rows', 3);
    clampParams();
    setParam('cliff_end', 4);
    createCanvas();
    stop_and_reset();
  });
  addButton(controls, 'Large', () => {
    setParam('cols', 10);
    setParam('rows', 4);
    clampParams();
    setParam('cliff_end', 8);
    createCanvas();
    stop_and_reset();
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
  controls.appendChild(stats);
  controls.appendChild(small);
  for (const [id, param] of Object.entries(paramData)) {
    addInput(boxes, id, param);
  }

  app.appendChild(heading);
  app.appendChild(controls);
  app.appendChild(boxes);
  createCanvas();
  app.appendChild(srctext);
}

function createCanvas() {
  if (canvas) {
    canvas.remove();
  }
  canvas = (document.createElement('canvas') as HTMLCanvasElement);
  canvas.width  = params.cols * TILE;
  canvas.height = params.rows * TILE;
  boxes.after(canvas);
  ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
}

function addInput(controls: HTMLDivElement, id: string, param: param) {
  controls.appendChild(createNumberInput(id, param))
}

function createNumberInput(id: string, param: param) {
  const [label, value, min, max] = param;
  params[id] = value;
  const wrapper = document.createElement('span');
  wrapper.style.margin = '10px';

  const lbl = document.createElement('label');
  lbl.textContent = ' ' + label + ' ';
  lbl.htmlFor = id;

  const input = document.createElement('input');
  input.type = 'number';
  input.id = id;
  input.min = '' + min;
  input.max = '' + max;
  // input.step = step ?? 1;
  input.value = '' + value;

  input.addEventListener('change', () => {
    const val = parseFloat(input.value);
    if (!isNaN(val)) {
      params[input.id] = val;
    }
    console.log(input.id, val);
    if (['rows', 'cols'].includes(input.id)) {
      console.log('changing size');
      clampParams();
      createCanvas();
      stop_and_reset();
    }
  })

  wrapper.appendChild(lbl);
  wrapper.appendChild(input);
  return wrapper;
}

function clampParams() {
  clamp('init_y', 'rows');
  clamp('goal_y', 'rows');
  clamp('goal_x', 'cols');
}

function clamp(param: string, limit: string) {
  if (params[param] >= params[limit]) {
    setParam(param, params[limit] - 1);
  }
}

function setParam(param: string, value: number) {
  params[param] = value;
  (document.getElementById(param) as HTMLInputElement).value = '' + value;
}

function addButton(controls: HTMLDivElement, text: string, f: () => void) {
  const b = document.createElement('button');
  b.textContent = text;
  b.addEventListener('click', f);
  controls.appendChild(b);
}

function log(...args: any[]): void {
  if (logging) {
    console.log(...args);
  }
}

initdom();
reset_qTable_and_state();
scheduleNext();
