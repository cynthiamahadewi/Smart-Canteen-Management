# Smart Canteen Management
The Smart Canteen Management System is a Flask-based web application designed to
reduce food waste and optimize canteen operations. By leveraging the M5 Deep Learning
Model, the system analyzes photos of leftover plates to calculate consumption ratios and
adjust serving sizes dynamically.

## 1. Install dependencies
```python
pip install flask torch torchvision openpyxl chartjs
```

## 2. Configure paths
Update dataset.EXCEL_PATH in app.py to point to your data_original.xlsx
## 3. Run the application
```python
python app.py
```