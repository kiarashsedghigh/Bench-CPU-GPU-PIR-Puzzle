# ⚡ QPADL: Post-Quantum Private Spectrum Access

> **Post-Quantum Secure • Privacy-Preserving • Location-Verified • DoS-Resilient**

<p align="center">
  <img src="https://img.shields.io/badge/crypto-post--quantum-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/security-verified-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/performance-GPU--accelerated-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-research--prototype-brightgreen?style=for-the-badge" />
</p>

---

## 📄 Paper Reference

This repository contains the **benchmarking** of the cryptographic components used in the paper:

> **QPADL: Post-Quantum Private Spectrum Access with Verified Location and DoS Resilience**  
> **Authors:** Saleh Darzi, Saif Eddine Nouma, Kiarash Sedghighadikolaei, Attila Altay Yavuz  
> **Published in:** *IEEE (Year / Venue TBD)*  
> **Contact:** ✉️ salehdarzi@usf.edu · saifeddinenouma@usf.edu · kiarashs@usf.edu · attilaayavuz@usf.edu

🔗 [ArXiv / DOI link – coming soon]

---

## 📝 Abstract

With advances in **wireless communication** and growing **spectrum scarcity**, Spectrum Access Systems (SASs) offer an opportunistic solution but face significant **security challenges**.

- 📍 **Location disclosure** and transmission details leak user privacy.
- 🎭 SAS is vulnerable to **spoofing attacks** by malicious users.
- 🚨 Database operations are prone to **Denial-of-Service (DoS) attacks**.
- ⚛️ Quantum computing amplifies these threats further.

---

## 🌐 What is QPADL?

**QPADL** is the **first post-quantum secure framework** that simultaneously ensures:

✅ **Privacy** — via SAS-tailored **Private Information Retrieval (PIR)**  
✅ **Anonymity** — via a **post-quantum variant of Tor**  
✅ **Location Verification** — through advanced **signature constructions**  
✅ **DoS Resilience** — using **client puzzles** + **rate-limiting techniques**

⚡ Designed for **large-scale spectrum access systems**, QPADL is **efficient, secure, and future-proof** against quantum threats.

---

## 🚀 Key Contributions

- 🔒 **Post-Quantum PIR** for location privacy
- 🕵️ **PQ-Tor**: Anonymous communication layer resistant to quantum attacks
- 📍 **Cryptographic location verification** with advanced signatures
- 🛡️ **DoS defenses** combining puzzles & rate limiting
- ⚙️ **GPU-accelerated performance** for scalability

---

✨ *This implementation is part of ongoing research. Contributions, feedback, and collaboration are welcome!*






## 📦 Repository Contents

- `src/` — Core source code of QPADL framework.
- `include/` — Header files.
- `tests/` — Unit and integration tests.
- `benchmarks/` — Benchmarking scripts and performance evaluation.
- `examples/` — Minimal usage examples and demos.
- `docs/` — Additional documentation and protocol descriptions.

---

## ⚙️ System Requirements

### 🖥️ Hardware
QPADL was evaluated on the following experimental setup:

- **CPU:** Intel Core i9-11900K @ 3.50 GHz
- **Memory:** 64 GiB RAM
- **Storage:** 1 TB SSD
- **OS:** Ubuntu 22.04.4 LTS
- **GPU (for acceleration):**
  - NVIDIA GeForce GTX 3060
  - 3584 CUDA cores
  - 12 GB GDDR6 VRAM
  - 360 GB/s memory bandwidth

> ⚡ QPADL supports both CPU-bounded execution (with AVX + OpenMP optimizations) and GPU-accelerated performance.

---

### 📚 Software & Libraries

- **Programming Languages:**
  - C/C++ (core implementation)
  - Python (supporting scripts & evaluation)

- **Cryptographic Libraries:**
  - [`percy++`](https://github.com/brianhe/percy) — multi-server PIR components
  - [`liboqs`](https://github.com/open-quantum-safe/liboqs) — post-quantum secure primitives
  - [`OpenSSL`](https://www.openssl.org/) — standard crypto ops (hash functions, RNG, etc.)
  - [`NTL`](https://libntl.org/) — lattice-based puzzle support
  - [`LRS`](https://github.com/xyz/LRS) — ring signature implementation
  - [`hashcash-tree`](https://github.com/xyz/hashcash-tree) — hash-based puzzle

- **Database:**
  - SQLite (for spectrum access database construction)

- **Parallelization:**
  - AVX instructions (CPU optimization)
  - OpenMP (multi-threaded execution)
  - CUDA (GPU acceleration)

---

## 🚀 Build Instructions
Each QPADL instantiation and puzzle generation module has its own dedicated folder inside the `src/` directory. 
Within these folders, you will find the corresponding **bash scripts** required to build and run the project.  

