import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <h2>Cliff Walking Reinforcement Learning</h2>
    <div class="controls">
        Algorithm: 
        <select id="algoSelect">
            <option value="qlearning">Q-Learning (Off-policy)</option>
            <option value="sarsa">SARSA (On-policy)</option>
            <option value="montecarlo">Monte Carlo (Every-visit)</option>
        </select>
        <button onclick="resetSim()">Reset & Run</button>
        <div class="stats" id="stats">Episode: 0 | Steps: 0 | Reward: 0</div>
        <small>Blue = Agent | Red = Cliff | Green = Goal | Yellow arrows = Policy</small>
    </div>
    <canvas id="gridCanvas" width="600" height="240"></canvas>
`
