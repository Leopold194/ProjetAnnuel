use std::fmt::Display;

/// Various utility functions for matrix operations and other calculations.

/// Multiplies two vectors element-wise.
pub fn vecvecmul(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() != b.len() {
        panic!("Vector dimensions do not match for multiplication.");
    }

    let mut result = vec![0.0; a.len()];

    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }

    return result
}

/// Multiplies two matrices.
pub fn matmatmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

/// Multiplies a matrix by a vector.
pub fn matvecmul(a: &Vec<Vec<f64>>,b:&Vec<f64>)->Vec<f64>{
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

/// Transposes a matrix.
pub fn transpose(matrix:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transposed = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

/// Inverts a matrix using Gaussian elimination.
pub fn inverse(mut matrix: Vec<Vec<f64>>, call: &str) -> Vec<Vec<f64>> {
    // Inversion de matrice, copie une fois en mémoire la matrice (pour avoir les deux en meme temps)

    let mut result = vec![vec![0.0; matrix.len()]; matrix.len()];
    let mut pivot:f64;
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
        
        println!("Pivot: {:?}", pivot);
        print_matrix(&matrix);
        
        if pivot==0.0{
            if call=="Moore-Penrose"{
                println!("La matrice possède des colonnes linéairements dépendantes");
                pivot = 1.0;
            }
            //panic!("La matrice n'est pas inversible");
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

pub fn print_matrix<T: Display>(matrix: &Vec<Vec<T>>) {
    for row in matrix {
        for item in row {
            print!("{}\t", item);
        }
        println!();
    }
}

// pub fn calc_determinant(matrice: Vec<Vec<f64>>) -> f64 {
//     //Utilisation pivot de Gauss pour calculer le déterminant.
//     //Modifications sur la matrice d'origine, donc faut pas passer le paramètre par référence.
//     //nvm je crée une copie au début comme ca on est bons.

//     let mut matrix: Vec<Vec<f64>> = matrice.clone();
    
//     if matrix.len() != matrix[0].len() {
//         panic!("Matrix is not square.");
//     }

//     let mut det:f64=1.0;
//     let n = matrix.len();

//     for i in 0..n{
//         // Recherche du pivot

//         if matrix[i][i] == 0.0 {
//             let mut found = false;
//             for k in i+1..n{
//                 if matrix[k][i] != 0.0{
//                     matrix.swap(i, k);
//                     det *=-1.0;
//                     found = true;
//                     break;
//                 }
//             }
            
//             // If no non-zero element is found, the determinant is zero.    
//             if found==false{
//                 return 0.0;
//             }

//         }
//         //application du pivot
//         for j in i+1..n{
//             let ratio = matrix[j][i] / matrix[i][i];
//             for k in i..n{
//                 matrix[j][k] -= ratio * matrix[i][k];
//             }
//         }

//     }
//     //calcul déterminant
//     for i in 0..n{
//         det *= matrix[i][i];
//     }
    
//     det
// }

