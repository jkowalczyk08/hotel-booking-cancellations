# Hotel Booking Cancellations

EDA and prediction models for hotel cancellations based on the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) on Kaggle.

## Project Structure

```
hotel-booking-cancellations/
├── data/           # Directory for dataset files
├── notebooks/      # Jupyter notebooks for analysis and modeling
├── main.py         # Main entry point (if applicable)
├── pyproject.toml  # Project configuration and dependencies
└── README.md       # Project documentation
```

## Getting Started

### Prerequisites

- **Git**: Ensure Git is installed on your system.
- **uv**: This project uses `uv` for dependency management. You can install it via:
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

### Dataset Setup

1.  Download the `hotel_bookings.csv` file from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
2.  Place the `hotel_bookings.csv` file inside the `data/` directory.

### Usage

To run the Jupyter notebooks:

1.  Activate the environment and launch Jupyter Lab (or Notebook):
    ```bash
    uv run jupyter lab
    ```
2.  Open `notebooks/01_eda_and_preprocessing.ipynb` to start exploring the data.