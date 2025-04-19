use std::vec;

use pyo3::prelude::*;
use rand::Rng;

struct LinearModel {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    label_map: Option<(String, String)>,
}



enum Labels {
    Str(Vec<String>),
    Float(Vec<f64>),
}

impl From<Vec<&str>> for Labels {
    fn from(labels: Vec<&str>) -> Self {
        Labels::Str(labels.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<Vec<f64>> for Labels {
    fn from(values: Vec<f64>) -> Self {
        Labels::Float(values)
    }
}

impl From<Vec<i32>> for Labels {
    fn from(values: Vec<i32>) -> Self {
        Labels::Float(values.into_iter().map(|v| v as f64).collect())
    }
}

struct MLP {
    npl: Vec<usize>, 
    weights: Vec<Vec<Vec<f64>>>,  
    l: usize, //layers    
    x:Vec<Vec<f64>>,
    deltas:Vec<Vec<f64>>  
}


impl MLP {
    fn new(npl: Vec<usize>) -> Self {
        let d = npl.clone();
        let l = d.len() - 1;
        let mut rng = rand::thread_rng();
    
        let mut weights = vec![vec![]];
    
        for layer in 1..=l {
            let mut layer_weights = Vec::new();
            
            let mut bias_weights = Vec::with_capacity(d[layer] + 1);
            bias_weights.push(0.0);
            for _ in 1..=d[layer] {
                bias_weights.push(rng.gen_range(-1.0..1.0));
            }
            layer_weights.push(bias_weights);
    
            for _ in 0..d[layer-1] {
                let mut neuron_weights = Vec::with_capacity(d[layer] + 1);
                neuron_weights.push(0.0);
                for _ in 1..=d[layer] {
                    neuron_weights.push(rng.gen_range(-1.0..1.0));
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

        MLP { npl, weights, l, x, deltas }
    }

    fn propagate(&mut self, inputs: Vec<f64>, is_classification: bool) {
        for j in 1..=self.npl[0]{
            self.x[0][j] = inputs[j-1];
        }
        for i in 1..=self.l{
            for j in 0..=self.npl[i]{
                let mut total: f64 = 0.0;
                for k in 0..=self.npl[i-1]{
                    total += self.weights[i][j][k]*self.x[i-1][j];
                }
                if (is_classification || i!=self.l){
                    total = tanh(total);
                }
                self.x[i][j] = total;
            }
        }
    }

    fn train(&mut self, all_inputs: &Vec<Vec<f64>>,all_outputs:&Vec<Vec<f64>>,epochs:usize,alpha:f64,is_classification: bool){
        for _ in 0..epochs{
            let mut k: usize= rand::thread_rng().gen_range(0..all_inputs.len());
            let sample_inputs:Vec<f64> = all_inputs[k].clone();
            let sample_outputs: Vec<f64> = all_outputs[k].clone();
            self.propagate(sample_inputs, is_classification);

            for j in 1..=self.npl[self.l]{
                self.deltas[self.l][j]= self.x[self.l][j]-sample_outputs[j-1];

                if is_classification{
                    self.deltas[self.l][j] *= (1 - self.X[self.L][j].powi(2));
                }
            }
            for i in 
        }
    }
}



impl LinearModel {
    fn new<T: Into<Labels>>(x: Vec<Vec<f64>>, y: T) -> Self {
        let mut label_map = None;

        let labels_norm = match y.into() {
            Labels::Str(labels) => {
                let first_label = labels[0].clone();
                let second_label = labels.iter().find(|l| **l != first_label)
                    .unwrap_or(&first_label).clone();
                label_map = Some((first_label.clone(), second_label.clone()));

                labels
                    .iter()
                    .map(|label| if *label == first_label { 1.0 } else { -1.0 })
                    .collect()
            }
            Labels::Float(values) => {
                let mut uniq: Vec<f64> = values.clone();
                uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
                uniq.dedup();

                if uniq.len() == 2 {
                    label_map = Some((uniq[1].to_string(), uniq[0].to_string())); // 1.0 -> uniq[1], -1.0 -> uniq[0]
                    values
                        .iter()
                        .map(|v| if *v == uniq[1] { 1.0 } else { -1.0 })
                        .collect()
                } else {
                    values
                }
            }
        };

        let mut rng = rand::thread_rng();
        let mut weights = (0..x[0].len()).map(|_| rng.gen_range(-1.0..1.0)).collect::<Vec<f64>>();
        weights.push(0.0); // biais

        LinearModel {
            x,
            y: labels_norm,
            weights,
            label_map,
        }
    }

    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            let mut rng = rand::thread_rng();
            let i = rng.gen_range(0..self.x.len());

            let random_x = self.x[i].clone();
            let prediction = self.predict_value(random_x.clone());
            let error = self.y[i] - prediction;

            for j in 0..self.weights.len() - 1 {
                self.weights[j] += learning_rate * error * random_x[j];
            }
            let last_index = self.weights.len() - 1; 
            self.weights[last_index] += learning_rate * error;
        }

        // println!("Weights: {:?}", self.weights);
    }

    fn train_regression(&mut self) {
        let mut x = self.x.clone();

        // Ajout du biais
        for i in 0..x.len(){
            x[i].push(1.0);
        }
        let y = self.y.clone(); 

        let xt: Vec<Vec<f64>> = transpose(&x);

        let xtx: Vec<Vec<f64>> = matmatmul(&xt, &x);
        let xtx_inv: Vec<Vec<f64>> = inverse(xtx.clone());
        let xtx_inv_xt: Vec<Vec<f64>> = matmatmul(&xtx_inv, &xt);
        self.weights = matvecmul(&xtx_inv_xt, &y);
        
        println!("Weights: {:?}", self.weights);
    }


    fn predict_value(&self, x: Vec<f64>) -> f64 {
        let mut sum = 0.0;
        let mut x_with_bias = x.clone();
        x_with_bias.push(1.0);

        for i in 0..x_with_bias.len() {
            sum += x_with_bias[i] * self.weights[i];
        }

        if sum > 0.0 { 1.0 } else { -1.0 }
    }

    fn predict(&self, x: Vec<f64>) -> String {
        let prediction = self.predict_value(x);

        if let Some((pos, neg)) = &self.label_map {
            if prediction > 0.0 {
                pos.clone()
            } else {
                neg.clone()
            }
        } else {
            prediction.to_string()
        }
    }
}

fn tanh(x: f64) -> f64 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

fn matmatmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if a[0].len() != b.len() {
        panic!("Matrix dimensions do not match for multiplication.");
    }
    let mut result = vec![vec![0.0; b[0].len()]; a.len()];
    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result
}

fn matvecmul(a: &Vec<Vec<f64>>,b:&Vec<f64>)->Vec<f64>{
    if a[0].len() != b.len() {
        panic!("Matrix and vector dimensions do not match for multiplication.");
    }
    let mut result = vec![0.0; a.len()];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i] += a[i][j] * b[j];
        }
    }
    return result
}

fn transpose(matrix:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transposed = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

fn inverse(mut matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    // Inversion de matrice, copie une fois en mémoire la matrice (pour avoir les deux en meme temps)

    let mut result = vec![vec![0.0; matrix.len()]; matrix.len()];
    let mut pivot:f64=0.0;

    for i in 0..matrix.len() {
        result[i][i] = 1.0;
    }
 
    for i in 0..matrix.len(){
        //récupération du pivot
        pivot = matrix[i][i];
        for j in i..matrix.len(){
            if matrix[j][i].abs()> pivot.abs(){
                pivot = matrix[j][i];
                matrix.swap(i, j);
                result.swap(i, j);
            }
        }

        if pivot==0.0{
            panic!("La matrice n'est pas inversible");
        }

        // normalisation ligne
        matrix[i].iter_mut().for_each(|x| *x /= pivot);
        result[i].iter_mut().for_each(|x| *x /= pivot);
        
        //normalisation colonne
        for j in 0..matrix.len(){
            if j!=i{
                let ratio = matrix[j][i];
                for k in 0..matrix[0].len(){
                    matrix[j][k] -= ratio * matrix[i][k];
                    result[j][k] -= ratio * result[i][k];
                }
            }
        }    
    }
    result
}

fn calc_determinant(matrice: Vec<Vec<f64>>) -> f64 {
    //Utilisation pivot de Gauss pour calculer le déterminant.
    //Modifications sur la matrice d'origine, donc faut pas passer le paramètre par référence.
    //nvm je crée une copie au début comme ca on est bons.

    let mut matrix: Vec<Vec<f64>> = matrice.clone();
    
    if matrix.len() != matrix[0].len() {
        panic!("Matrix is not square.");
    }

    let mut det:f64=1.0;
    let n = matrix.len();

    for i in 0..n{
        // Recherche du pivot

        if matrix[i][i] == 0.0 {
            let mut found = false;
            for k in i+1..n{
                if matrix[k][i] != 0.0{
                    matrix.swap(i, k);
                    det *=-1.0;
                    found = true;
                    break;
                }
            }
            
            // If no non-zero element is found, the determinant is zero.    
            if found==false{
                return 0.0;
            }

        }
        //application du pivot
        for j in i+1..n{
            let ratio = matrix[j][i] / matrix[i][i];
            for k in i..n{
                matrix[j][k] -= ratio * matrix[i][k];
            }
        }

    }
    //calcul déterminant
    for i in 0..n{
        det *= matrix[i][i];
    }
    
    det
}


fn main() {
    let mlp = MLP::new(vec![2,3, 1]);
    println!("MLP: {:?}", mlp.weights);
}

fn demo_simple_regression() {
    println!("\n=== Régression simple (y = 2*x) ===");
    
    let x = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
    ];
    let y = vec![2.0, 4.0, 6.0, 8.0];
    
    let mut model = LinearModel::new(x, y);
    model.train_regression();
    
    println!("Poids obtenus: {:?}", model.weights);
    println!("Poids attendus: ~[2.0]");
}

fn demo_multivariate_regression() {
    println!("\n=== Régression multivariée (y = 1 + 2*x1 + 3*x2) ===");
    
    let x = vec![
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![2.0, 2.0],
    ];
    let y = vec![6.0, 9.0, 8.0, 11.0];
    
    let mut model = LinearModel::new(x, y);
    model.train_regression();
    
    println!("Poids obtenus: {:?}", model.weights);
    println!("Poids attendus: ~[2.0, 3.0] (plus un biais)");
}



/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn projetannuel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}