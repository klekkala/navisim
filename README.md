### Project Setup
#### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/klekkala/navisim.git
   cd navisim
   ```

2. **(Optional) Create and activate a virtual environment**

   **Using venv (Python built-in):**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```
   Using conda(if preferred):
   ```bash
   conda create --name your-env-name python=3.x
   conda activate your-env-name
   ```
   
3. Install required packages
   ```
   pip install -r requirements.txt
   ```
---

### How to Use
1. Download `sequence_graph.gpickle` and `database.tar` from [this link](https://drive.google.com/drive/folders/1mGaTTTblhbVnN_i5AivSQU1ZWyZJIY87?usp=drive_link) into the `assets/` folder.

2. Extract the contents of `database.tar` using the command below:

   ```bash
   tar -xvf database.tar -C assets/
