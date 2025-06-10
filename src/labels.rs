use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub enum LabelsEnum {
    Str(Vec<String>),
    Float(Vec<f64>),
}

#[pyfunction]
pub fn string_labels(labels: Vec<String>) -> LabelsEnum {
    LabelsEnum::Str(labels)
}

#[pyfunction]
pub fn float_labels(labels: Vec<f64>) -> LabelsEnum {
    LabelsEnum::Float(labels)
}

impl From<Vec<&str>> for LabelsEnum {
    fn from(labels: Vec<&str>) -> Self {
        LabelsEnum::Str(
            labels
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        )
    }
}

impl From<Vec<f64>> for LabelsEnum {
    fn from(values: Vec<f64>) -> Self {
        LabelsEnum::Float(values)
    }
}

impl From<Vec<i32>> for LabelsEnum {
    fn from(values: Vec<i32>) -> Self {
        LabelsEnum::Float(
            values
                .into_iter()
                .map(|v| v as f64)
                .collect()
        )
    }
}