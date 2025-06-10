use crate::utils;
use crate::labels;
use crate::linear_abstract::LinearModelAbstract;

use ordered_float::OrderedFloat;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{ PyString, PyFloat, PyDict };
use rand::Rng;

#[pyclass]
pub struct LinearModel {
    #[pyo3(get, set)]
    pub weights: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub loss: Vec<f64>,
    #[pyo3(get, set)]
    pub x: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub y: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub labels_original: Option<labels::LabelsEnum>,
    #[pyo3(get, set)]
    pub model_type: String,
    pub label_map_str: Option<HashMap<String, usize>>,
    pub label_map_float: Option<HashMap<OrderedFloat<f64>, usize>>,
    #[pyo3(get, set)]
    pub num_classes: usize,
}

#[pymethods]
impl LinearModel {
    #[new]
    #[pyo3(signature = (x, y, labels_original=None))]
    pub fn new(x: Vec<Vec<f64>>, y: Vec<Vec<f64>>, labels_original: Option<labels::LabelsEnum>) -> Self {
        // let dim = x[0].len() + 1;
        // let weights = (0..dim).map(|_| rand::random::<f64>()).collect();

        LinearModel {
            weights: vec![vec![]],
            loss: vec![],
            x,
            y,
            labels_original,
            model_type: String::new(),
            label_map_str: None,
            label_map_float: None,
            num_classes: 0,
        }
    }

    pub fn train_classification(&mut self, py: Python<'_>, epochs: usize, learning_rate: f64, algo: &str) {
        <Self as LinearModelAbstract>::train_classification(self, py, epochs, learning_rate, algo)
    }

    pub fn train_regression(&mut self, py: Python<'_>) {
        <Self as LinearModelAbstract>::train_regression(self, py)
    }

    pub fn predict(&self, py: Python<'_>, x: Vec<f64>, return_label: Option<bool>) -> PyResult<PyObject> {
        <Self as LinearModelAbstract>::predict(self, py, x, return_label)
    }

    // pub fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
    //     <Self as LinearModelAbstract>::predict_proba(self, py, x)
    // }
}

impl LinearModelAbstract for LinearModel {
    fn weights(&self) -> Vec<Vec<f64>> { self.weights.clone() }
    fn loss(&self) -> Vec<f64> { self.loss.clone() }
    fn num_classes(&self) -> usize { self.num_classes.clone() }
    fn get_x(&self) -> &Vec<Vec<f64>> { &self.x }
    fn get_y(&self) -> &Vec<Vec<f64>> { &self.y }
    fn get_labels_original(&self) -> Option<&labels::LabelsEnum> { self.labels_original.as_ref() }
    fn get_model_type(&self) -> &str { &self.model_type }
    fn set_model_type(&mut self, model_type: String) { self.model_type = model_type; }
    fn get_label_map_str(&self) -> Option<HashMap<String, usize>> { self.label_map_str.clone() }
    fn get_label_map_float(&self) -> Option<HashMap<OrderedFloat<f64>, usize>> { self.label_map_float.clone() }
    fn set_label_map_str(&mut self, map: Option<HashMap<String, usize>>) { self.label_map_str = map; }
    fn set_label_map_float(&mut self, map: Option<HashMap<OrderedFloat<f64>, usize>>) { self.label_map_float = map; }
    fn set_weights(&mut self, w: Vec<Vec<f64>>) { self.weights = w; }
    fn set_loss(&mut self, l: Vec<f64>) { self.loss = l; }
    fn get_num_classes(&self) -> usize { self.num_classes }
    fn set_num_classes(&mut self, num_classes: usize) {self.num_classes = num_classes; }
}
