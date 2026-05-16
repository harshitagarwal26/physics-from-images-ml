# How to Run (Pendulum & Dumbbell) — Step‑by‑Step

This guide is written for someone who has **not** used a terminal or Python
before. Follow it top to bottom. Every line in a grey box is a command: copy
it, paste it into the Terminal, and press **Return**. Each command has a
one‑line explanation above it.

It is written for **macOS**. There are two experiments:

- **Part A — Pendulum** (LoRA delta‑V model)
- **Part B — Dumbbell**

Do **Part 0 (one‑time setup) first**, then either part.

---

## Part 0 — One‑time setup (do this once)

### 0.1 Open the Terminal

Press `Cmd` + `Space`, type **Terminal**, press **Return**. A window with a
text prompt opens. That is where every command below goes.

### 0.2 Go into the project folder

This moves the Terminal into the project folder you downloaded. The easiest
way: type `cd ` (with a space), then **drag the project folder from Finder onto
the Terminal window** (this pastes its path), then press **Return**. It will
look like the example below — your path may differ:

```
cd ~/Downloads/Lagrangian_caVAE-clean
```

Check you are in the right place — this should list files like `examples`,
`datasets`, `requirements.txt`:

```
ls
```

### 0.3 Install Python 3.9

This project needs **Python 3.9 specifically** (newer versions will not work
with its older libraries).

1. Open this page in a browser: <https://www.python.org/downloads/release/python-3913/>
2. Scroll to **Files**, download **“macOS 64‑bit universal2 installer”**.
3. Open the downloaded `.pkg` file and click through the installer (Continue →
   Agree → Install).

Confirm it installed (this should print `Python 3.9.x`):

```
python3.9 --version
```

### 0.4 Create a private workspace for the libraries (“virtual environment”)

This makes an isolated folder named `venv` so this project’s libraries don’t
interfere with anything else on the computer.

```
python3.9 -m venv venv
```

### 0.5 Turn the workspace on (“activate”)

**You must run this command every time you open a new Terminal window before
running anything else.** After it runs, the prompt will start with `(venv)`.

```
source venv/bin/activate
```

### 0.6 Install all required libraries

This reads `requirements.txt` and installs everything (PyTorch, Jupyter, etc.).
It downloads a lot and can take **10–30 minutes**. Wait until the prompt
returns.

```
pip install --upgrade pip
pip install -r requirements.txt
```

Setup is done. In future sessions you only repeat **0.2** (`cd ...`) and
**0.5** (`source venv/bin/activate`).

---

## Part A — Pendulum pipeline (LoRA delta‑V)

This trains a small “correction” model on top of a pre‑trained pendulum model,
then you open a notebook to look at the results.

> Make sure the prompt shows `(venv)`. If not, run `source venv/bin/activate`
> first.

### A.1 Train the pendulum LoRA model

This command does the training. It uses a provided pre‑trained base model and a
provided dataset. **Training is slow — it can run for several hours** on a
laptop. You can leave it running; it prints progress numbers. When it is done
the Terminal prompt comes back.

```
python examples/pend_delta_v_lora_r8_trainer.py \
  --pretrained_ckpt results/pend/pend-lag-cavae-T_p=4-epoch=983-step=7871.ckpt \
  --data_path datasets/pendulum-gym-image-dataset-train-reverse-angle-perturbed-10sin6_372.pkl \
  --max_epochs 1000 --T_pred 4
```

The trained model is saved automatically into the folder
`logs/pend-delta-v-lora-r8/`.

### A.2 Open the analysis notebook

This starts **Jupyter**, which opens in your web browser:

```
jupyter notebook
```

A browser tab opens showing the project files. In that tab:

1. Click the folder **`results`**, then **`pend`**.
2. Click the file **`8_deltaV_lora.ipynb`**. It opens as a notebook.
3. In the top menu choose **Run ▸ Run All Cells**.
4. Scroll down — the plots appear under the cells.

This notebook compares the LoRA‑corrected models against the true dynamics and
draws the result plots. It is already set up to read the **provided result
checkpoints** (for example `results/pend/10sin6_372_lora8.ckpt`), so it works
out of the box without you editing anything.

*(Optional — only if you want to view the model you just trained in A.1
instead of the provided one: in the notebook find the cell that sets a
checkpoint path ending in `_lora8.ckpt`, and change it to the newest `.ckpt`
file inside `logs/pend-delta-v-lora-r8/`. If unsure, skip this — the provided
checkpoint already shows the intended result.)*

When finished, go back to the Terminal and press `Ctrl` + `C` to stop Jupyter.

---

## Part B — Dumbbell pipeline

This trains the dumbbell model, then you open a notebook to look at the
reconstruction and the learned physics.

> Make sure the prompt shows `(venv)`. If not, run `source venv/bin/activate`
> first.

### B.1 Train the dumbbell model

**Important:** you must include `--data_name dumbbell-rigid-dataset-grayscale.pkl`
exactly as written (this version of the model expects the black‑and‑white
dataset). Training is slow — **allow several hours**.

```
python examples/dumbbell_lag_cavae_trainer.py \
  --max_epochs 1000 --T_pred 4 \
  --data_name dumbbell-rigid-dataset-grayscale.pkl
```

The trained model is saved automatically into `logs/dumbbell-lag-cavae/`.

> Honest note: the dumbbell model is research‑in‑progress. A fresh training run
> may not perfectly reproduce the saved result. For that reason the analysis
> notebook below is pre‑set to a **known‑good saved checkpoint** so you always
> see a sensible result. Running the training in B.1 is still worth doing to
> see the process.

### B.2 Open the analysis notebook

Start Jupyter (skip this if it is already running from Part A):

```
jupyter notebook
```

In the browser tab:

1. Click the folder **`examples`**.
2. Click the file **`visualize_dumbbell_cavae.ipynb`**.
3. In the top menu choose **Run ▸ Run All Cells**.
4. Scroll through — you will see, in order: ground‑truth vs reconstructed
   dumbbell images, encoder accuracy scatter plots, the multi‑step prediction,
   and the learned potential / mass‑matrix plots.

This notebook is **already pointed at the correct checkpoint**
(`results/dumbbell/grayscale_baseline_v22/last_grayscale_v22.ckpt`) and the
correct dataset (`datasets/dumbbell-rigid-dataset-grayscale.pkl`). You do not
need to edit anything — just Run All Cells.

*(Optional — to instead view the model you trained in B.1: open the notebook,
find the first code cell that sets `CKPT_PATH = ...`, and replace the path with
the newest `.ckpt` file inside `logs/dumbbell-lag-cavae/`. If unsure, skip
this.)*

When finished, go to the Terminal and press `Ctrl` + `C` to stop Jupyter.

---

## Quick reference

| Task | Command / file |
|---|---|
| Activate the workspace (every new Terminal) | `source venv/bin/activate` |
| Pendulum analysis notebook | `results/pend/8_deltaV_lora.ipynb` |
| Dumbbell analysis notebook | `examples/visualize_dumbbell_cavae.ipynb` |
| Start Jupyter | `jupyter notebook` |
| Stop Jupyter | `Ctrl` + `C` in the Terminal |

## Common problems

- **`command not found: python3.9`** — Python 3.9 isn’t installed; redo step 0.3.
- **Commands fail and the prompt has no `(venv)`** — run `source venv/bin/activate`.
- **`pip install` is slow / seems stuck** — it is downloading large files; give
  it up to 30 minutes before worrying.
- **Jupyter didn’t open a browser** — copy the `http://localhost:8888/...` link
  the Terminal printed and paste it into a browser.
- **A notebook cell shows a red error** — use **Kernel ▸ Restart Kernel**, then
  **Run ▸ Run All Cells** again from the top.
- **Training seems to never finish** — these models train for many hours on a
  laptop; this is expected. You can stop early with `Ctrl` + `C`; partial
  checkpoints are still saved in the `logs/` folder.
