use crate::utils;
use crate::labels;
use crate::lloyd::lloyd;
use crate::linear_abstract::LinearModelAbstract;

use ordered_float::OrderedFloat;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{ PyString, PyFloat, PyDict };
use rand::Rng;
use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct RBF {
    model_name:String,
    #[pyo3(get, set)]
    pub weights: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub train_loss: Vec<f64>,
    #[pyo3(get, set)]
    pub test_loss: Vec<f64>,
    #[pyo3(get, set)]
    pub train_accuracy: Vec<f64>,
    #[pyo3(get, set)]
    pub test_accuracy: Vec<f64>,
    #[pyo3(get, set)]
    pub x: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub y: labels::LabelsEnum,
    #[pyo3(get)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub model_type: String,
    #[pyo3(get, set)]
    pub centers: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub gamma: f64,
    pub label_map_str: Option<HashMap<String, usize>>,
    pub label_map_float: Option<HashMap<OrderedFloat<f64>, usize>>,
    #[pyo3(get, set)]
    pub num_classes: usize,
}

#[pymethods]
impl RBF {
    #[new]
    #[pyo3(signature = (x, y, gamma, k, seed=None))]
    pub fn new(py: Python<'_>, x: Vec<Vec<f64>>, y: labels::LabelsEnum, gamma: f64, k: i32, seed: Option<u64>) -> Self {
        
        if k > x.len() as i32 {
            panic!("You cannot have more centers than inputs.")
        }

        let centers = lloyd(py, x.clone(), k, 2.22e-16, seed);
        
        let mut phi: Vec<Vec<f64>> = Vec::with_capacity(x.len());
        for row in x.clone() {
            phi.push(utils::convert_x_to_phi(row, centers.clone(), gamma))
        }

        // let dim = phi[0].len() + 1;
        // let weights = (0..dim).map(|_| rand::random::<f64>()).collect();
        
        RBF {
            model_name:String::from("RBF"),
            weights: vec![vec![]],
            train_loss: vec![],
            test_loss: vec![],
            train_accuracy: vec![],
            test_accuracy: vec![],
            x: phi,
            y,
            seed,
            model_type: String::new(),
            centers,
            gamma,
            label_map_str: None,
            label_map_float: None,
            num_classes: 0,
        }
    }

    #[pyo3(signature = (epochs, learning_rate, algo=None, x_test=None, y_test=None))]
    pub fn train_classification(&mut self, py: Python<'_>, epochs: usize, learning_rate: f64, algo: Option<String>, x_test: Option<Vec<Vec<f64>>>, y_test: Option<labels::LabelsEnum>) {
        let algo = algo.as_deref().unwrap_or("gradient-descent");

        let phi = if let Some(ref x_test_data) = x_test {
            let mut phi_vec = Vec::with_capacity(x_test_data.len());
            for row in x_test_data {
                phi_vec.push(utils::convert_x_to_phi(row.clone(), self.centers.clone(), self.gamma));
            }
            Some(phi_vec)
        } else {
            None
        };

        <Self as LinearModelAbstract>::train_classification(self, py, epochs, learning_rate, algo, phi, y_test)
    }

    pub fn train_regression(&mut self) {
        <Self as LinearModelAbstract>::train_regression(self)
    }

    pub fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        <Self as LinearModelAbstract>::predict(self, py, utils::convert_x_to_phi(x, self.centers.clone(), self.gamma))
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: RBF = serde_json::from_reader(file).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }

    // pub fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
    //     <Self as LinearModelAbstract>::predict_proba(self, py, utils::convert_x_to_phi(x, self.centers.clone(), self.gamma))
    // }
}

impl LinearModelAbstract for RBF {
    fn weights(&self) -> Vec<Vec<f64>> { self.weights.clone() }
    fn train_loss(&self) -> Vec<f64> { self.train_loss.clone() }
    fn test_loss(&self) -> Vec<f64> { self.test_loss.clone() }
    fn train_accuracy(&self) -> Vec<f64> { self.train_accuracy.clone() }
    fn test_accuracy(&self) -> Vec<f64> { self.test_accuracy.clone() }
    fn seed(&self) -> Option<u64> { self.seed.clone() }
    fn num_classes(&self) -> usize { self.num_classes.clone() }
    fn get_x(&self) -> &Vec<Vec<f64>> { &self.x }
    fn get_y(&self) -> &labels::LabelsEnum { &self.y }
    fn get_model_type(&self) -> &str { &self.model_type }
    fn set_model_type(&mut self, model_type: String) { self.model_type = model_type; }
    fn get_label_map_str(&self) -> Option<HashMap<String, usize>> { self.label_map_str.clone() }
    fn get_label_map_float(&self) -> Option<HashMap<OrderedFloat<f64>, usize>> { self.label_map_float.clone() }
    fn set_label_map_str(&mut self, map: Option<HashMap<String, usize>>) { self.label_map_str = map; }
    fn set_label_map_float(&mut self, map: Option<HashMap<OrderedFloat<f64>, usize>>) { self.label_map_float = map; }
    fn set_weights(&mut self, w: Vec<Vec<f64>>) { self.weights = w; }
    fn set_train_loss(&mut self, l: Vec<f64>) { self.train_loss = l; }
    fn set_test_loss(&mut self, l: Vec<f64>) { self.test_loss = l; }
    fn set_train_accuracy(&mut self, a: Vec<f64>) { self.train_accuracy = a; }
    fn set_test_accuracy(&mut self, a: Vec<f64>) { self.test_accuracy = a; }
    fn get_num_classes(&self) -> usize { self.num_classes }
    fn set_num_classes(&mut self, num_classes: usize) {self.num_classes = num_classes; }
}
