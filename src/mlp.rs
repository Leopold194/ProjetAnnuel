use pyo3::prelude::*;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::weighted::WeightedIndex;
use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct MLP {
    model_type: String,
    npl: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,  
    l: usize,
    x:Vec<Vec<f64>>,
    deltas:Vec<Vec<f64>>,
    #[pyo3(get)]
    train_loss:Vec<f64>,
    #[pyo3(get)]
    test_loss:Vec<f64>,
}


#[pymethods]
impl MLP {

    #[new]
    fn new(npl: Vec<usize>, seed: usize) -> Self {
        let mut r = StdRng::seed_from_u64(seed as u64);
        let d = npl.clone();
        let l = d.len() - 1;
        let mut rng = rand::rng();
    
        let mut weights = vec![vec![]];
    
        for layer in 1..=l {
            let mut layer_weights = Vec::new();
            
            let mut bias_weights = Vec::with_capacity(d[layer] + 1);
            bias_weights.push(0.0);
            for _ in 1..=d[layer] {
                bias_weights.push(r.random_range(-1.0..1.0));
            }
            layer_weights.push(bias_weights);
    
            for _ in 0..d[layer-1] {
                let mut neuron_weights = Vec::with_capacity(d[layer] + 1);
                neuron_weights.push(0.0);
                for _ in 1..=d[layer] {
                    neuron_weights.push(r.random_range(-1.0..1.0));
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
            model_type:String::from("MLP"),
            npl, 
            weights, 
            l, 
            x, 
            deltas,
            train_loss:Vec::new(),
            test_loss:Vec::new()
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
    pub fn train_loss(&self) -> Vec<f64> {
        self.train_loss.clone()
    }

    #[getter]
    pub fn test_loss(&self)->Vec<f64>{
        self.test_loss.clone()
    }

     pub fn train(
        &mut self,
        X_train: Vec<Vec<f64>>,
        y_train: Vec<Vec<f64>>,
        X_test: Vec<Vec<f64>>,
        y_test: Vec<Vec<f64>>,
        iterations: usize,
        alpha: f64,
        is_classification: bool,
        seed: u64,
    ) {
        self.train_loss.clear();
        self.test_loss.clear();
        let mut rng = StdRng::seed_from_u64(seed);
        let L = self.npl.len() - 1;
        let num_outputs = self.npl[L] as f64;

        for _ in 0..iterations {
            let mut total_train = 0.0;
            for (xt, yt) in X_train.iter().zip(y_train.iter()) {
                self.propagate(xt.to_vec(), is_classification);
                for j in 1..=self.npl[L] {
                    let y_hat = self.x[L][j];
                    let y = yt[j - 1];
                    if is_classification {
                        let eps = 1e-8;
                        let y_hat_clamped = y_hat.max(eps).min(1.0 - eps);
                        total_train += -y * y_hat_clamped.ln() - (1.0 - y) * (1.0 - y_hat_clamped).ln();
                    } else {
                        total_train += 0.5 * (y_hat - y).powi(2);
                    }
                }
            }
            self.train_loss.push(total_train / (X_train.len() as f64 * num_outputs));

            let mut total_test = 0.0;
            for (xt, yt) in X_test.iter().zip(y_test.iter()) {
                self.propagate(xt.to_vec(), is_classification);
                for j in 1..=self.npl[L] {
                    let y_hat = self.x[L][j];
                    let y = yt[j - 1];
                    if is_classification {
                        let eps = 1e-8;
                        let y_hat_clamped = y_hat.max(eps).min(1.0 - eps);
                        total_test += -y * y_hat_clamped.ln() - (1.0 - y) * (1.0 - y_hat_clamped).ln();
                    } else {
                        total_test += 0.5 * (y_hat - y).powi(2);
                    }
                }
            }
            self.test_loss.push(total_test / (X_test.len() as f64 * num_outputs));

            let idx = rng.gen_range(0..X_train.len());
            let inputs = &X_train[idx];
            let targets = &y_train[idx];
            self.propagate(inputs.to_vec(), is_classification);

            let idx = rng.gen_range(0..X_train.len());
            let inputs = &X_train[idx];
            let targets = &y_train[idx];

            self.propagate(inputs.to_vec(), is_classification);

            for j in 1..=self.npl[L] {
                let y_hat = self.x[L][j];
                let error = y_hat - targets[j - 1];
                self.deltas[L][j] = if is_classification {
                    error * (1.0 - y_hat.powi(2))
                } else {
                    error
                };
            }

            for l in (1..L).rev() {
                for i in 1..=self.npl[l] {
                    let mut sum = 0.0;
                    for k in 1..=self.npl[l + 1] {
                        sum += self.weights[l + 1][i][k] * self.deltas[l + 1][k];
                    }
                    let x_i = self.x[l][i];
                    self.deltas[l][i] = sum * (1.0 - x_i.powi(2));
                }
            }

            for l in 1..=L {
                for i in 0..=self.npl[l - 1] {
                    for j in 1..=self.npl[l] {
                        self.weights[l][i][j] -= alpha * self.x[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
    }

    fn predict(&mut self, inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.x[self.l][1..].to_vec()
    }
    
    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: MLP = serde_json::from_reader(file).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }
}