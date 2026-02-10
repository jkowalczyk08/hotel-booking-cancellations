# Hotel Booking Cancellations

EDA and prediction models for hotel cancellations based on the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) on Kaggle.

## Models

Models are in separate notebooks. We used:
- Logistic regression
- Random forest
- Decision tree
- kNN
- Ada boost
- XG boost

For each model we used 3 different preprocessing functions defined in src/model_utils.py.

## Project Structure

```
hotel-booking-cancellations/
├── data/               # Raw, cleaned and split dataset files (.csv, .parquet)
├── notebooks/          # Jupyter notebooks for analysis and modeling
├── src/                # Shared Python utility modules
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # Project documentation
```

## Markdown cells
Some cells are in Polish for university presentation.

## Getting Started

### Prerequisites

- **Git**: Ensure Git is installed on your system.
- **uv**: This project uses [uv](https://docs.astral.sh/uv/) for dependency management. You can install it via:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jkowalczyk08/hotel-booking-cancellations.git
    cd hotel-booking-cancellations
    ```

2.  **Install dependencies:**
    Sync the environment to install all required packages defined in `pyproject.toml`.
    ```bash
    uv sync
    ```
