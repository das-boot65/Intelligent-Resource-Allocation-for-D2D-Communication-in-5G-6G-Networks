#  Intelligent Resource Allocation for D2D Communication in 5G/6G Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)]()
[![RL](https://img.shields.io/badge/Reinforcement%20Learning-PPO-green)]()
[![5G/6G](https://img.shields.io/badge/Network-5G%2F6G-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()

---

##  Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Simulation Setup](#simulation-setup)
- [Algorithm](#algorithm)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contributors](#contributors)

---

##  Overview
This project presents a **reinforcement learning**–driven approach using PPO to optimize **device-to-device (D2D) communication** in **5G/6G networks**. Designed for simulation in **edge computing environments**, it improves **throughput**, **energy efficiency**, and **adaptability**, offering a scalable solution for next-gen wireless systems.

---

##  Objectives
- **Optimize** spectrum and power allocation in D2D communication  
- **Minimize** interference while ensuring high spectral efficiency  
- **Improve** energy efficiency and provide robust QoS (Quality of Service)  
- **Demonstrate** scalability of AI-driven methods in 5G/6G networks  

---

##  System Architecture

**Network Model**  
- Simulates a cellular base station with multiple D2D pairs sharing spectrum.

**Resource Allocation Framework**  
- Multi-agent PPO (Proximal Policy Optimization) dynamically assigns channels and power.  
- **State**: wireless channel quality, interference, traffic load  
- **Action**: channel selection, power level assignment  

**Reward Function**  
- Designed to boost **throughput**, reduce **power use**, and ensure **fairness**.

---

##  Simulation Setup
- **Language**: Python  
- **Libraries**: TensorFlow / PyTorch, NumPy, Pandas, Matplotlib, Seaborn  
- **Environment**: Simulated 5G/6G and edge-computing infrastructure  
- **Metrics**:
  - Throughput (Mbps)
  - Energy efficiency (bits/Joule)
  - Fairness
  - Adaptability under varied conditions

---

##  Algorithm
**Proximal Policy Optimization (PPO)**  
- A stable, policy-gradient RL algorithm  
- Optimized for training multi-agent scenarios over iterative simulation episodes

---

##  Results
- PPO outperforms baseline (random/heuristic) allocation methods  
- Achieves **higher throughput** and **spectral efficiency**, with improved energy metrics  
- Demonstrates strong **adaptability** in dynamic network and traffic conditions

![Learning rate accross models](results/learning_rate_comparision.png)


---

##  How to Run

### Prerequisites
- Python 3.8+  
- pip

### Installation
```bash
git clone https://github.com/your-username/AI-D2D-Resource-Allocation.git
cd AI-D2D-Resource-Allocation
pip install -r requirements.txt
