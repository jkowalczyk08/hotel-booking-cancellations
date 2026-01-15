# Hotel Booking Cancellations

EDA and prediction models for hotel cancellations based on the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) on Kaggle.

## Project Structure

```
hotel-booking-cancellations/
├── data/           # Directory for dataset files
├── notebooks/      # Jupyter notebooks for analysis and modeling
├── pyproject.toml  # Project configuration and dependencies
└── README.md       # Project documentation
```

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
