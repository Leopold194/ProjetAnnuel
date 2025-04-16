use pyo3::prelude::*;
use rand::Rng;

struct LinearModel {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
}

enum Labels {
    Str(Vec<String>),
    Float(Vec<f64>),
}

// Permet d'utiliser directement Vec<&str> sans intervention de l'utilisateur
impl From<Vec<&str>> for Labels {
    fn from(labels: Vec<&str>) -> Self {
        Labels::Str(labels.into_iter().map(|s| s.to_string()).collect())
    }
}

// Permet d'utiliser directement Vec<f64> sans intervention de l'utilisateur
impl From<Vec<f64>> for Labels {
    fn from(values: Vec<f64>) -> Self {
        Labels::Float(values)
    }
}

impl LinearModel {
    fn new<T: Into<Labels>>(x: Vec<Vec<f64>>, y: T) -> Self {

        let labels_norm = match y.into() {
            Labels::Str(labels) => {
                let first_label = &labels[0];
                labels
                    .iter()
                    .map(|label| {
                        if *label == *first_label {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect()
            }
            Labels::Float(values) => values,
        };

        let mut weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..x[0].len() {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        weights.push(0.0);
        
        LinearModel {
            x: x,
            y: labels_norm,
            weights: weights,
        }
    }

    fn train_classification(&mut self, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            let mut rng = rand::thread_rng();
            let i = rng.gen_range(0..self.x.len());

            let random_x = self.x[i].clone();
            let prediction = self.predict(random_x.clone());
            let error = self.y[i] - prediction;

            for j in 0..self.weights.len() - 1 {
                self.weights[j] += learning_rate * error * random_x[j];
            }
            let last_index = self.weights.len() - 1;
            self.weights[last_index] += learning_rate * error * 1.0; // Biais                
        }

        println!("Weights: {:?}", self.weights);
    }

    fn train_regression(&mut self) {
        //let xt: Vec<Vec<f64>> = transpose(self.x);
        //self.weights = matvecmul(matmatmul(inverse(matmatmul(xt,self.x)),xt),self.y);
        println!("Weights: {:?}", self.weights);
    }


    fn predict(&self, x: Vec<f64>) -> f64 {
        let mut sum: f64 = 0.0;

        let mut x_with_bias = x.clone();
        x_with_bias.push(1.0); // Ajout du biais (1.0)

        for i in 0..x_with_bias.len() {
            sum += x_with_bias[i] * self.weights[i];
        }

        if sum > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}


fn matmatmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

fn matvecmul(a: Vec<Vec<f64>>,b:Vec<f64>)->Vec<f64>{
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

fn transpose(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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


fn print_matrix(matrix: &[Vec<f64>]) {
    for row in matrix {
        println!("{:?}", row);
    }
}

fn main() {
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
    let y = vec!["true", "true", "false", "true"];
    // let y = vec![1.0, 1.0, -1.0, 1.0];
    println!("x: {:?}", x);
    println!("y: {:?}", y);
    let mut model = LinearModel::new(x, y);
    model.train_classification(1000000, 0.001);

    println!("Prediction: {:?}", model.predict(vec![1.0, 0.0]));
    println!("Prediction: {:?}", model.predict(vec![0.0, 1.0]));
    println!("Prediction: {:?}", model.predict(vec![0.0, 0.0]));
    println!("Prediction: {:?}", model.predict(vec![1.0, 1.0]));
    
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ];
    let b = vec![
        vec![7.0, 8.0],
        vec![9.0, 10.0],
        vec![11.0, 12.0],
    ];
    let mat_result = matmatmul(a, b);
    println!("Matrix multiplication result: {:?}", mat_result);

    // Test matvecmul
    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let vector = vec![2.0, 1.0, 3.0];
    let vec_result = matvecmul(matrix, vector);
    println!("Matrix-vector multiplication result: {:?}", vec_result);

    // Example 2: 3x3 matrix
    let matrix_3x3 = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0]
    ];
    println!("3x3 matrix: {:?}", matrix_3x3);
    println!("Determinant: {}\n", calc_determinant(matrix_3x3.clone())); // Expected: 0 (singular matrix)

    // Example 3: Another 3x3 matrix
    let matrix_3x3_alt = vec![
        vec![2.0, -1.0, 3.0],
        vec![0.0, 4.0, -2.0],
        vec![1.0, 0.0, 5.0]
    ];
    println!("Alternative 3x3 matrix: {:?}", matrix_3x3_alt);
    println!("Determinant: {}", calc_determinant(matrix_3x3_alt)); // Expected: 30

    // Matrice 2x2
    let matrix_2x2 = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0]
    ];
    
    println!("Matrice 2x2 originale:");
    print_matrix(&matrix_2x2);
    
    let inverse_2x2 = inverse(matrix_2x2);
    println!("\nMatrice inverse 2x2:");
    print_matrix(&inverse_2x2);
    
    // Matrice 3x3
    let matrix_3x3 = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 1.0, 4.0],
        vec![5.0, 6.0, 0.0]
    ];
    
    println!("\nMatrice 3x3 originale:");
    print_matrix(&matrix_3x3);
    
    let inverse_3x3 = inverse(matrix_3x3);
    println!("\nMatrice inverse 3x3:");
    print_matrix(&inverse_3x3);
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
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}