use pyo3::prelude::*;
use rand::Rng;
use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct MLP {
    npl: Vec<usize>, // neurons per layer
    weights: Vec<Vec<Vec<f64>>>,  
    l: usize, //layers    
    x:Vec<Vec<f64>>,
    deltas:Vec<Vec<f64>>,
    #[pyo3(get)]
    loss:Vec<f64>
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
        for _ in 0..epochs {
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

            // if epoch % 100 == 0 {

            //     let msg = if is_classification {
            //         format!("Epoch {} – BCE: {:.6}", epoch, loss)
            //     } else {
            //         format!("Epoch {} – MSE: {:.6}", epoch, loss)
            //     };
        

            //     Python::with_gil(|py| {
            //         py_print(py, &msg).expect("failed to print to Python stdout");
            //     });
            // }


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