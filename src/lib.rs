mod utils;

use pyo3::prelude::*;
use pyo3::types::{ PyString, PyFloat, PyList };
use rand::Rng;



#[pyclass]
struct LinearModel {
    x: Vec<Vec<f64>>,
    y: LabelsEnum,
    weights: Vec<f64>,
    label_map_str: Option<(String, String)>,
    label_map_float: Option<(f64, f64)>,
    model_type: String,
    loss: Vec<f64>,
}

#[pyclass]
struct MLP {
    npl: Vec<usize>, // neurons per layer
    weights: Vec<Vec<Vec<f64>>>,  
    l: usize, //layers    
    x:Vec<Vec<f64>>,
    deltas:Vec<Vec<f64>>,
    #[pyo3(get)]
    loss:Vec<f64>
}

#[pyclass]
#[derive(Clone)]
enum LabelsEnum {
    Str(Vec<String>),
    Float(Vec<f64>),
}

#[pyfunction]
fn string_labels(labels: Vec<String>) -> LabelsEnum {
    LabelsEnum::Str(labels)
}

#[pyfunction]
fn float_labels(labels: Vec<f64>) -> LabelsEnum {
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

#[allow(dead_code)]
fn py_print(py: Python<'_>, msg: &str) -> PyResult<()> {
    let builtins = PyModule::import(py, "builtins")?;
    builtins.call_method1("print", (msg,))?;  // call the “print” attribute with one arg
    Ok(())
}

#[pymethods]
impl MLP {

    #[new]
    fn new(npl: Vec<usize>) -> Self {
        let d = npl.clone();
        let l = d.len() - 1;
        let mut rng = rand::rng();
    
        let mut weights = vec![vec![]];
    
        for layer in 1..=l {
            let mut layer_weights = Vec::new();
            
            let mut bias_weights = Vec::with_capacity(d[layer] + 1);
            bias_weights.push(0.0);
            for _ in 1..=d[layer] {
                bias_weights.push(rng.random_range(-1.0..1.0));
            }
            layer_weights.push(bias_weights);
    
            for _ in 0..d[layer-1] {
                let mut neuron_weights = Vec::with_capacity(d[layer] + 1);
                neuron_weights.push(0.0);
                for _ in 1..=d[layer] {
                    neuron_weights.push(rng.random_range(-1.0..1.0));
                }
                layer_weights.push(neuron_weights);
            }
    
            weights.push(layer_weights);
        }
        
        let mut x = Vec::with_capacity(l + 1);
        let mut deltas = Vec::with_capacity(l + 1);
        
        for &neurons in d.iter() {
            let mut x_layer = vec![1.0]; 
            x_layer.extend(vec![0.0; neurons]); 
            
            let delta_layer = vec![0.0; neurons + 1];             
            x.push(x_layer);
            deltas.push(delta_layer);
        }

        MLP { 
            npl, 
            weights, 
            l, 
            x, 
            deltas,
            loss:Vec::new()
        }
    }

    fn propagate(&mut self, inputs: Vec<f64>, is_classification: bool) {
        for j in 1..=self.npl[0] {
            self.x[0][j] = inputs[j-1];
        }
        
        for i in 1..=self.l {
            for j in 1..=self.npl[i] { 
                let mut total: f64 = 0.0;
                for k in 0..=self.npl[i-1] { 
                    total += self.weights[i][k][j] * self.x[i-1][k];
                }
                
                if is_classification || i != self.l {
                    self.x[i][j] = total.tanh();
                } else {
                    self.x[i][j] = total;
                }
            }
        }
    }

    #[getter]
    pub fn loss(&self) -> Vec<f64> {
        self.loss.clone()
    }

    fn train(&mut self, all_inputs: Vec<Vec<f64>>, all_outputs: Vec<Vec<f64>>, epochs: usize, alpha: f64, is_classification: bool) {
        for epoch in 0..epochs {
            let k = rand::rng().random_range(0..all_inputs.len());
            let sample_inputs = all_inputs[k].clone();
            let sample_outputs = all_outputs[k].clone();
            
            self.propagate(sample_inputs, is_classification);


            let mut loss = 0.0;

            for j in 1..=self.npl[self.l] {
                let y_hat = self.x[self.l][j];
                let y = sample_outputs[j - 1];
    
                if is_classification {
                    let eps = 1e-8;
                    let y_hat_clipped = y_hat.max(eps).min(1.0 - eps);
                    loss += -y * y_hat_clipped.ln() - (1.0 - y) * (1.0 - y_hat_clipped).ln();

                } else {
                    loss += 0.5 * (y_hat - y).powi(2);

                }
            }

            let num_outputs:f64 = self.npl[self.l] as f64;
            self.loss.push(loss/num_outputs);

            if epoch % 100 == 0 {

                let msg = if is_classification {
                    format!("Epoch {} – BCE: {:.6}", epoch, loss)
                } else {
                    format!("Epoch {} – MSE: {:.6}", epoch, loss)
                };
        

                Python::with_gil(|py| {
                    py_print(py, &msg).expect("failed to print to Python stdout");
                });
            }


            for j in 1..=self.npl[self.l] {
                let error = self.x[self.l][j] - sample_outputs[j-1];
                if is_classification {
                    self.deltas[self.l][j] = error * (1.0 - self.x[self.l][j].powi(2));
                } else {
                    self.deltas[self.l][j] = error;
                }
            }

            for i in (1..self.l).rev() {
                for j in 1..=self.npl[i] {
                    let mut sum = 0.0;
                    for k in 1..=self.npl[i+1] {
                        sum += self.weights[i+1][j][k] * self.deltas[i+1][k];
                    }
                    self.deltas[i][j] = sum * (1.0 - self.x[i][j].powi(2));
                }
            }

            for i in 1..=self.l {
                for j in 0..=self.npl[i-1] {
                    for k in 1..=self.npl[i] {
                        self.weights[i][j][k] -= alpha * self.x[i-1][j] * self.deltas[i][k];
                    }
                }
            }
        }
    }

    // fn train_classification(&mut self, all_inputs: Vec<Vec<f64>>, all_outputs: Vec<Vec<f64>>, epochs: usize, alpha: f64)

    fn predict(&mut self, inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.x[self.l][1..].to_vec()
    }
}

#[pymethods]
impl LinearModel {
    
    /// Creates a new LinearModel instance.
    /// The `x` parameter is a 2D vector representing the input features,
    /// and the `y` parameter can be a vector of strings or floats representing the labels.
    /// The function normalizes the labels to -1 and 1 for classification tasks.
    
    #[new]
    fn new(x: Vec<Vec<f64>>, y: LabelsEnum) -> Self {
        let mut rng = rand::rng();
        let mut weights = (0..x[0].len()).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<f64>>();
        weights.push(0.0); // biais

        LinearModel {
            x,
            y,
            weights,
            label_map_str: None,
            label_map_float: None,
            model_type: "".to_string(),
            loss: vec![],
        }
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[getter]
    fn loss(&self) -> Vec<f64> {
        self.loss.clone()
    }

    fn calc_log_loss(&mut self, y: Vec<f64>, sig_val: Vec<f64>) -> f64 {
        let mut loss = 0.0;

        for i in 0..sig_val.len() {
            loss += y[i] * sig_val[i].ln() + (1.0 - y[i]) * (1.0 - sig_val[i]).ln();
        }

        (- 1.0 / sig_val.len() as f64) * loss
    }

    fn calc_gradients(&mut self, sig_val: Vec<f64>, x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
        let n = y.len();
        let m = x[0].len();

        let mut weights_grad = vec![0.0; x[0].len()];
        let mut bias_grad = 0.0;

        let mut diff;

        for i in 0..n {
            diff = sig_val[i] - y[i];

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

    fn update_weights(&mut self, weights_grad: Vec<f64>, learning_rate: f64){
        for i in 0..self.weights.len() {
            self.weights[i] -= weights_grad[i] * learning_rate;
        }
    }

    /// Trains the model for a specified number of epochs using the given learning rate.
    /// The training process involves adjusting the weights based on the input features and labels.
    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
        
        self.model_type = "classification".to_string();
        
        let mut label_map_str = None;
        let mut label_map_float = None;

        let labels_norm = match &self.y {
            LabelsEnum::Str(labels) => {
                let first_label = labels[0].clone();
                let second_label = labels
                    .iter()
                    .find(|l| **l != first_label)
                    .unwrap_or(&first_label)
                    .clone();
                label_map_str = Some((first_label.clone(), second_label.clone()));

                labels
                    .iter()
                    .map(|label| if *label == first_label { 1.0 } else { 0.0 })
                    .collect()
            }
            LabelsEnum::Float(values) => {
                let mut uniq: Vec<f64> = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();

                if uniq.len() == 2 {
                    label_map_float = Some((uniq[1], uniq[0]));
                    values
                        .iter()
                        .map(|v| if *v == uniq[1] { 1.0 } else { 0.0 })
                        .collect()
                } else {
                    values.clone()
                }
            }
        };

        self.label_map_str = label_map_str;
        self.label_map_float = label_map_float;

        let mut loss = Vec::with_capacity(labels_norm.len());
        let mut a;
        let mut l;
        let mut weights_grad;

        for _ in 0..epochs {
            a = self.model(self.x.clone());
            l = self.calc_log_loss(labels_norm.clone(), a.clone());
            loss.push(l);
            weights_grad = self.calc_gradients(a.clone(), self.x.clone(), labels_norm.clone());
            self.update_weights(weights_grad, learning_rate);
        }

        self.loss = loss;
    }

    /// Trains the model using linear regression.
    /// The training process involves calculating the weights using the normal equation.
    fn train_regression(&mut self) {

        self.model_type = "regression".to_string();

        let mut x = self.x.clone();

        // Ajout du biais
        for i in 0..x.len(){
            x[i].push(1.0);
        }
        let y = self.y.clone();
        let y: Vec<f64> = match y {
            LabelsEnum::Str(_labels) => {
                panic!("La regression ne fonctionne pas avec des labels de type String")
            }
            LabelsEnum::Float(values) => values,
        };

        let xt: Vec<Vec<f64>> = utils::transpose(&x);

        let xtx: Vec<Vec<f64>> = utils::matmatmul(&xt, &x);
        let call: &str = "Moore-Penrose";
        let xtx_inv: Vec<Vec<f64>> = utils::inverse(xtx.clone(), &call);
        let xtx_inv_xt: Vec<Vec<f64>> = utils::matmatmul(&xtx_inv, &xt);
        self.weights = utils::matvecmul(&xtx_inv_xt, &y);
    }

    /// Predicts the output for a given input vector `x` using the trained model.
    /// The function calculates the weighted sum of the input features and applies the activation function.
    fn model(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let mut weights_without_bias = self.weights.clone();
        weights_without_bias.pop();
        let mut z = utils::matvecmul(&x, &weights_without_bias);

        let bias = self.weights[self.weights.len() - 1];
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

    fn predict_classification(&self, x: Vec<f64>) -> f64 {
        let val = self.model(vec![x])[0];
        match val {
            tmp if tmp >= 0.5 => {
                return 1.0
            }
            _ => {
                return 0.0
            }
        }
    }

    fn predict_regression(&self, x: Vec<f64>) -> f64 {
        let mut x_with_bias: Vec<f64> = x.clone();
        x_with_bias.push(1.0);
        let y = utils::vecvecmul(&self.weights,&x_with_bias);
        let mut value:f64=0.0;

        for i in 0..y.len(){
            value+=y[i];
        }

        value
    }
    
    pub fn predict(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<PyObject> {
        match self.model_type.as_str() {
            "regression" => {
                let value = self.predict_regression(x);
                Ok(PyFloat::new(py, value).into())
            }
            _ => {
                let prediction = self.predict_classification(x);

                if let Some((pos, neg)) = &self.label_map_str {
                    let label_str = if prediction > 0.0 { pos } else { neg };
                    Ok(PyString::new(py, label_str).into())
                } else {
                    let (pos, neg) = self
                        .label_map_float
                        .as_ref()
                        .expect("label_map_float should never be None if label_map_str is None");
                    let label_float = if prediction > 0.0 { pos } else { neg };
                    Ok(PyFloat::new(py, *label_float).into())
                }
                
            }
        }
    }
}

#[allow(dead_code)]
#[pyfunction]
fn accuracy_score(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {
    let y_true = y_true.downcast_bound::<PyList>(py)?;
    let y_pred = y_pred.downcast_bound::<PyList>(py)?;

    if y_true.len() != y_pred.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_true et y_pred doivent avoir la même longueur",
        ));
    }

    let mut correct = 0;

    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        if let (Ok(list_a), Ok(list_b)) = (a.downcast::<PyList>(), b.downcast::<PyList>()) {
            if list_a.len() != list_b.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Les sous-listes doivent avoir la même taille",
                ));
            }

            let vec_a: Vec<f64> = list_a.extract()?;
            let vec_b: Vec<f64> = list_b.extract()?;

            let argmax_a = vec_a
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);
            let argmax_b = vec_b
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);

            if argmax_a == argmax_b {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<f64>(), b.extract::<f64>()) {
            if (va - vb).abs() < 1e-6 {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<i64>(), b.extract::<i64>()) {
            if va == vb {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<&str>(), b.extract::<&str>()) {
            if va == vb {
                correct += 1;
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Types non supportés pour la comparaison",
            ));
        }
    }

    Ok(correct as f64 / y_true.len() as f64)
}

#[allow(dead_code)]
#[pyfunction]
fn mean_squared_error(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {

    let y_true = y_true.downcast_bound::<PyList>(py)?;
    let y_pred = y_pred.downcast_bound::<PyList>(py)?;

    if y_true.len() != y_pred.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_true et y_pred doivent avoir la même longueur",
        ));
    }

    let val_nb = y_true.len();
    let mut errors = Vec::with_capacity(val_nb);

    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        if let (Ok(va), Ok(vb)) = (a.extract::<f64>(), b.extract::<f64>()) {
            errors.push(va - vb);
        } else if let (Ok(va), Ok(vb)) = (a.extract::<i64>(), b.extract::<i64>()) {
            errors.push((va - vb) as f64);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Les types ne correspondent pas pour la comparaison (doivent être float ou int)",
            ));
        }
    }

    let mut squared_errors_sum: f64 = 0.0;

    for val in errors {
        squared_errors_sum += val * val;
    }

    Ok(squared_errors_sum / val_nb as f64)
}

#[allow(dead_code)]
#[pyfunction]
fn root_mean_squared_error(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {
    let mse = mean_squared_error(py, y_true, y_pred);
    
    Ok(mse?.sqrt())
}

pub fn main() {

}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn projetannuel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accuracy_score,m)?)?;
    m.add_function(wrap_pyfunction!(mean_squared_error,m)?)?;
    m.add_function(wrap_pyfunction!(root_mean_squared_error,m)?)?;

    m.add_class::<LinearModel>()?;
    m.add_class::<MLP>()?;

    m.add_function(wrap_pyfunction!(float_labels,m)?)?;
    m.add_function(wrap_pyfunction!(string_labels,m)?)?;

    Ok(())
}
