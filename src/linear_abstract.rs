use crate::utils;
use crate::labels;

use ordered_float::OrderedFloat;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyString};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub trait LinearModelAbstract {
    fn weights(&self) -> Vec<Vec<f64>>;

    fn train_loss(&self) -> Vec<f64>;
    fn test_loss(&self) -> Vec<f64>;

    fn train_accuracy(&self) -> Vec<f64>;
    fn test_accuracy(&self) -> Vec<f64>;

    fn num_classes(&self) -> usize;
    fn seed(&self) -> Option<u64>;

    fn get_x(&self) -> &Vec<Vec<f64>>;
    fn get_y(&self) -> &labels::LabelsEnum;
    fn get_model_type(&self) -> &str;
    fn set_model_type(&mut self, model_type: String);

    fn get_num_classes(&self) -> usize;
    fn set_num_classes(&mut self, num_classes: usize);

    fn get_label_map_str(&self) -> Option<HashMap<String, usize>>;
    fn get_label_map_float(&self) -> Option<HashMap<OrderedFloat<f64>, usize>>;
    fn set_label_map_str(&mut self, map: Option<HashMap<String, usize>>);
    fn set_label_map_float(&mut self, map: Option<HashMap<OrderedFloat<f64>, usize>>);

    fn set_weights(&mut self, w: Vec<Vec<f64>>);

    fn set_train_loss(&mut self, l: Vec<f64>);
    fn set_test_loss(&mut self, l: Vec<f64>);

    fn set_train_accuracy(&mut self, acc: Vec<f64>);
    fn set_test_accuracy(&mut self, acc: Vec<f64>);
    
    fn model(&self, py: Python<'_>, x: Vec<Vec<f64>>, idx: usize) -> Vec<f64> {
        let mut weights_without_bias = self.weights()[idx].clone();
        weights_without_bias.pop();

        let mut z = utils::matvecmul(&x, &weights_without_bias);

        let bias = self.weights()[idx].last().cloned().unwrap_or(0.0);
        for val in z.iter_mut() {
            *val += bias;
        }

        // activation function (sigmoid)
        let mut result = Vec::with_capacity(z.len());

        let eps = 1e-15;
        for i in 0..z.len() {
            let neg_x = -z[i];
            let sigmoid = 1.0 / (1.0 + neg_x.exp());
            // car si valeur trèèès grande de z, exp(-z) = 1 et si trèèès petit = 0
            let clamped = sigmoid.clamp(eps, 1.0 - eps);
            result.push(clamped);
        }
        result
    }

    fn calc_log_loss(&self, y: Vec<f64>, sig_val: Vec<f64>) -> f64 {
        let mut loss = 0.0;
        for i in 0..sig_val.len() {
            loss += y[i] * sig_val[i].ln() + (1.0 - y[i]) * (1.0 - sig_val[i]).ln();
        }
        -loss / sig_val.len() as f64
    }

    fn calc_gradients(&self, sig_val: Vec<f64>, x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
        let n = y.len();
        let m = x[0].len();
        let mut weights_grad = vec![0.0; m];
        let mut bias_grad = 0.0;

        for i in 0..n {
            let diff = sig_val[i] - y[i];
            for j in 0..m {
                weights_grad[j] += x[i][j] * diff;
            }
            bias_grad += diff;
        }

        for j in 0..m {
            weights_grad[j] /= n as f64;
        }
        bias_grad /= n as f64;

        weights_grad.push(bias_grad);
        weights_grad
    }

    fn update_weights(&mut self, grad: Vec<f64>, learning_rate: f64, idx: usize) -> Vec<f64> {
        let mut w = self.weights()[idx].clone();
        for i in 0..w.len() {
            w[i] -= grad[i] * learning_rate;
        }
        // self.set_weights(w);
        w
    }

    fn compute_accuracy(&mut self, y_true: Vec<f64>, y_pred: Vec<f64>) -> f64 {
        let mut correct = 0;
        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            if (*yt - *yp).abs() < 2.2e-16 {
                correct += 1;
            }
        }
        correct as f64 / y_true.len() as f64
    }

    fn train_classification(&mut self, py: Python<'_>, epochs: usize, learning_rate: f64, algo: &str, x_test: Option<Vec<Vec<f64>>>, y_test: Option<labels::LabelsEnum>) {
        self.set_model_type("classification".to_string());
        let mut rng_opt: Option<StdRng> = self.seed().map(StdRng::seed_from_u64);

        let y_labels = self.get_y().clone();
        let labels_norm: Vec<_> = match &y_labels {
            labels::LabelsEnum::Str(labels) => {
                // Créer mapping: label string → index
                let mut uniq = labels.clone();
                uniq.sort();
                uniq.dedup();
                let map: HashMap<String, usize> = uniq
                    .iter()
                    .enumerate()
                    .map(|(i, label)| (label.clone(), i))
                    .collect();

                // Stocker l'inverse: index → label
                self.set_label_map_str(Some(map.clone()));

                labels.iter().map(|l| map[l] as f64).collect()
            }
            labels::LabelsEnum::Float(values) => {
                let mut uniq = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();
                let map: HashMap<OrderedFloat<f64>, usize> = uniq
                    .iter()
                    .map(|v| OrderedFloat(*v))
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();

                self.set_label_map_float(Some(map.clone()));

                values.iter().map(|v| map[&OrderedFloat(*v)] as f64).collect()
            }
        };

        let labels_norm_test = if let (Some(x_test_data), Some(y_test_labels)) = (&x_test, &y_test) {
            Some(match y_test_labels {
                labels::LabelsEnum::Str(labels) => {
                    let map = self.get_label_map_str().expect("No label map for strings");
                    labels.iter().map(|l| map[l] as f64).collect::<Vec<f64>>()
                }
                labels::LabelsEnum::Float(values) => {
                    let map = self.get_label_map_float().expect("No label map for floats");
                    values.iter().map(|v| map[&OrderedFloat(*v)] as f64).collect::<Vec<f64>>()
                }
            })
        } else {
            None
        };

        let mut mean_losses = vec![0.0; epochs];
        let mut mean_losses_test = vec![0.0; epochs];

        let mut accuracy = vec![0.0; epochs];
        let mut accuracy_test = vec![0.0; epochs];

        let mut unique_classes = labels_norm.clone();
        unique_classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_classes.dedup();

        let dim = self.get_x()[0].len() + 1;
        let mut weights = Vec::with_capacity(unique_classes.len());

        if let Some(rng) = rng_opt.as_mut() {
            for _ in 0..unique_classes.len() {
                weights.push((0..dim).map(|_| rng.r#gen::<f64>()).collect());
            }
        } else {
            for _ in 0..unique_classes.len() {
                weights.push((0..dim).map(|_| rand::random::<f64>()).collect());
            }
        }

        self.set_weights(weights);
        self.set_num_classes(unique_classes.len());

        // Préparer les labels binaires pour toutes les classes
        let mut binary_labels: Vec<Vec<f64>> = Vec::new();
        let mut binary_labels_test: Vec<Option<Vec<f64>>> = Vec::new();

        for class_label in unique_classes.iter() {
            let binary_y: Vec<f64> = labels_norm.iter().map(|v| if v == class_label { 1.0 } else { 0.0 }).collect();
            binary_labels.push(binary_y);

            let binary_y_test: Option<Vec<f64>> = labels_norm_test.as_ref().map(|test_labels| {
                test_labels.iter().map(|v| if v == class_label { 1.0 } else { 0.0 }).collect()
            });
            binary_labels_test.push(binary_y_test);
        }

        // CHANGEMENT PRINCIPAL : Boucle par époque, puis par classe
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_loss_test = 0.0;

            // Entraîner chaque classe à cette époque
            for (idx, class_label) in unique_classes.iter().enumerate() {
                let binary_y = &binary_labels[idx];
                let binary_y_test = &binary_labels_test[idx];

                if algo == "gradient-descent" {
                    let a = self.model(py, self.get_x().clone(), idx);
                    let l = self.calc_log_loss(binary_y.clone(), a.clone());
                    epoch_loss += l;

                    let grad = self.calc_gradients(a.clone(), self.get_x().clone(), binary_y.clone());
                    let class_weights = self.update_weights(grad, learning_rate, idx);
                    let mut all_weights = self.weights();
                    all_weights[idx] = class_weights.clone();
                    self.set_weights(all_weights);

                    // Calculer la perte de test si disponible
                    if let (Some(x_test_data), Some(binary_y_test_vec)) = (&x_test, binary_y_test) {
                        let pred_test = self.model(py, x_test_data.clone(), idx);
                        let test_loss = self.calc_log_loss(binary_y_test_vec.clone(), pred_test.clone());
                        epoch_loss_test += test_loss;
                    }
                } else if algo == "rosenblatt" {
                    let i = if let Some(rng) = rng_opt.as_mut() {
                        rng.gen_range(0..self.get_x().len())
                    } else {
                        rand::thread_rng().gen_range(0..self.get_x().len())
                    };
                    let random_x = self.get_x()[i].clone();

                    let prediction = self.model(py, vec![random_x.clone()], idx);
                    let error = binary_y[i] - prediction[0];

                    let l = self.calc_log_loss(binary_y.clone(), prediction.clone());
                    epoch_loss += l;

                    let mut w = self.weights()[idx].clone();
                    for j in 0..w.len() - 1 {
                        w[j] += learning_rate * error * random_x[j];
                    }

                    let last_index = w.len() - 1;
                    w[last_index] += learning_rate * error;
                    
                    let mut all_weights = self.weights();
                    all_weights[idx] = w;
                    self.set_weights(all_weights);

                    // Calculer la perte de test si disponible
                    if let (Some(x_test_data), Some(binary_y_test_vec)) = (&x_test, binary_y_test) {
                        let pred_test = self.model(py, x_test_data.clone(), idx);
                        let test_loss = self.calc_log_loss(binary_y_test_vec.clone(), pred_test.clone());
                        epoch_loss_test += test_loss;
                    }
                } else {
                    panic!("This algorithm is not supported.")
                }
            }

            // Calculer l'accuracy multiclasse à cette époque
            let train_predictions = self.predict_class(py, self.get_x().clone());
            accuracy[epoch] = self.compute_accuracy(labels_norm.clone(), train_predictions);

            // Calculer l'accuracy de test si disponible
            if let (Some(x_test_data), Some(labels_norm_test_vec)) = (&x_test, &labels_norm_test) {
                let test_predictions = self.predict_class(py, x_test_data.clone());
                accuracy_test[epoch] = self.compute_accuracy(labels_norm_test_vec.clone(), test_predictions);
            }

            // Stocker les pertes moyennes
            mean_losses[epoch] = epoch_loss / self.get_num_classes() as f64;
            if x_test.is_some() {
                mean_losses_test[epoch] = epoch_loss_test / self.get_num_classes() as f64;
            }
        }

        self.set_train_loss(mean_losses);
        self.set_test_loss(mean_losses_test);
        self.set_train_accuracy(accuracy);
        self.set_test_accuracy(accuracy_test);
    }

    fn train_regression(&mut self) {
        self.set_model_type("regression".to_string());

        let mut x = self.get_x().clone();
        for row in &mut x {
            row.push(1.0);
        }

        let y = match self.get_y() {
            labels::LabelsEnum::Str(_) => panic!("Regression incompatible with String labels"),
            labels::LabelsEnum::Float(v) => v.clone(),
        };

        let xt = utils::transpose(&x);
        let xtx = utils::matmatmul(&xt, &x);
        let xtx_inv = utils::inverse(xtx, "Moore-Penrose");
        let xtx_inv_xt = utils::matmatmul(&xtx_inv, &xt);
        let weights = utils::matvecmul(&xtx_inv_xt, &y);
        self.set_weights(vec![weights]);
    }

    // fn predict_proba(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
    //     let proba = self.model(py, vec![x])[0];
    //     let result = PyDict::new(py);

    //     if let Some((pos, neg)) = self.get_label_map_str() {
    //         result.set_item(pos, proba)?;
    //         result.set_item(neg, 1.0 - proba)?;
    //     } else if let Some((pos, neg)) = self.get_label_map_float() {
    //         result.set_item(pos.to_string(), proba)?;
    //         result.set_item(neg.to_string(), 1.0 - proba)?;
    //     }

    //     Ok(result.into())
    // }

    fn predict_binary_classification(&self, py: Python<'_>, x: Vec<f64>) -> f64 {
        if self.model(py, vec![x], 0)[0] >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    fn predict_regression(&self, x: Vec<f64>) -> f64 {
        let mut x_bias = x;
        x_bias.push(1.0);
        utils::vecvecmul(&self.weights()[0], &x_bias).iter().sum()
    }

    fn predict_class(&self, py: Python<'_>, x_data: Vec<Vec<f64>>) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(x_data.len());
        
        for sample in x_data.iter() {
            // Utilise la même logique que votre fonction predict existante
            let classes_weights = self.weights();
            let num_classes = self.get_num_classes();

            let predicted_class_idx = if num_classes > 1 {
                let mut best_score = 0.0;
                let mut predicted_class_idx: usize = 0;
                
                for i in 0..classes_weights.len() {
                    let val = self.model(py, vec![sample.clone()], i)[0];
                    if val > best_score {
                        best_score = val;
                        predicted_class_idx = i;
                    }
                }
                predicted_class_idx
            } else {
                let prediction = self.predict_binary_classification(py, sample.clone());
                if prediction > 0.0 { 1 } else { 0 }
            };
            
            predictions.push(predicted_class_idx as f64);
        }
        
        predictions
    }

    fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        match self.get_model_type() {
            "regression" => Ok(PyFloat::new(py, self.predict_regression(x)).into()),
            "classification" => {
                let classes_weights = self.weights();
                let num_classes = self.get_num_classes();

                if num_classes > 1 {
                    let mut best_score = 0.0;
                    let mut predicted_class_idx: usize = 0;
                    let mut val: f64;

                    // utils::py_print(py, &format!("map : {:?}", self.get_label_map_str()));
                    
                    for i in 0..classes_weights.len() {
                        val = self.model(py, vec![x.clone()], i)[0];
                        // utils::py_print(py, &format!("val : {:?}", val));
                        if val > best_score {
                            best_score = val;
                            predicted_class_idx = i;
                        }
                    }

                    if let Some(map) = self.get_label_map_str() {
                        let label = map.iter()
                            .find(|&(_, &v)| v == predicted_class_idx)
                            .map(|(k, _)| k.clone())
                            .unwrap_or_else(|| predicted_class_idx.to_string());
                        Ok(PyString::new(py, &label).into())
                    } else if let Some(map) = self.get_label_map_float() {
                        let label = map.iter()
                            .find(|&(_, &v)| v == predicted_class_idx)
                            .map(|(k, _)| *k)
                            .unwrap_or(OrderedFloat(predicted_class_idx as f64));
                        Ok(PyFloat::new(py, *label).into())
                    } else {
                        Ok(PyFloat::new(py, predicted_class_idx as f64).into())
                    }

                } else {
                    let prediction = self.predict_binary_classification(py, x);
                    if let Some(map) = self.get_label_map_str() {
                        let label = map.iter()
                            .find(|&(_, &v)| v == if prediction > 0.0 { 1 } else { 0 })
                            .map(|(k, _)| k.clone())
                            .unwrap_or_else(|| prediction.to_string());
                        Ok(PyString::new(py, &label).into())
                    } else if let Some(map) = self.get_label_map_float() {
                        let label = map.iter()
                            .find(|&(_, &v)| v == if prediction > 0.0 { 1 } else { 0 })
                            .map(|(k, _)| *k)
                            .unwrap_or(OrderedFloat(prediction));
                        Ok(PyFloat::new(py, *label).into())
                    } else {
                        Ok(PyFloat::new(py, prediction).into())
                    }
                }
            },
            _ => panic!("Unknow model type"),
        }
    }
}
