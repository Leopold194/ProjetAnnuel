use pyo3::prelude::*;
use osqp::{CscMatrix, Problem, Settings};

#[pyclass]
#[derive(Clone)]
pub enum SVMKernelType {
    Linear(),
    Polynomial { degree: usize },
    RBF { gamma: f64 },
}

#[pyclass]
pub struct SVM {
    #[pyo3(get)]
    pub alpha: Vec<f64>,
    #[pyo3(get)]
    pub support_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub support_labels: Vec<f64>,
    #[pyo3(get)]
    pub bias: f64,
    #[pyo3(get)]
    pub kernel: SVMKernelType,
}

#[pymethods]
impl SVM {
    #[new]
    pub fn new(kernel: SVMKernelType) -> Self {
        SVM {
            alpha: vec![],
            support_vectors: vec![],
            support_labels: vec![],
            bias: 0.0,
            kernel,
        }
    }

    fn train(&mut self, x_py: Vec<Vec<f64>>, y_py: Vec<f64>) -> PyResult<()> {
        let n = x_py.len();
        let d = x_py[0].len();

        let mut q_dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let k = match &self.kernel {
                    SVMKernelType::Linear => dot(&x_py[i], &x_py[j]),
                    SVMKernelType::Polynomial { degree } => {
                        (1.0 + dot(&x_py[i], &x_py[j])).powi(*degree as i32)
                    }
                    SVMKernelType::RBF { gamma } => {
                        let dist2 = squared_distance(&x_py[i], &x_py[j]);
                        (-gamma * dist2).exp()
                    }
                };
                q_dense[i][j] = y_py[i] * y_py[j] * k;
            }
        }

        let (q_data, q_indices, q_indptr) = to_csc(&q_dense);
        let p = CscMatrix::new(n, n, q_indptr, q_indices, q_data)?;
        let q_vec = vec![-1.0; n];

        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        for j in 0..n {
            a_data.push(y_py[j]);        
            a_indices.push(0);

            a_data.push(1.0);            
            a_indices.push((j + 1) as i32);

            a_indptr.push(a_data.len() as i32);
        }

        let a = CscMatrix::new(n + 1, n, a_indptr, a_indices, a_data)?;
        let mut l = vec![0.0; n + 1];
        let mut u = vec![0.0; n + 1];
        for i in 1..=n {
            u[i] = std::f64::INFINITY;
        }

        let settings = Settings::default();
        let prob = Problem::new(p, &q_vec, a, &l, &u, &settings)?;
        let result = prob.solve();
        let alpha = result.x().unwrap().to_vec();

        self.alpha = vec![];
        self.support_vectors = vec![];
        self.support_labels = vec![];

        for i in 0..n {
            if alpha[i] > 1e-6 {
                self.alpha.push(alpha[i]);
                self.support_vectors.push(x_py[i].clone());
                self.support_labels.push(y_py[i]);
            }
        }

        if !self.alpha.is_empty() {
            let i = 0; 
            let sv = &self.support_vectors[i];
            let y = self.support_labels[i];
            let mut s = 0.0;
            for j in 0..self.alpha.len() {
                s += self.alpha[j] * self.support_labels[j]
                    * self.kernel_fn(&self.support_vectors[j], sv);
            }
            self.bias = y - s;
        }

        Ok(())
    }

    fn predict(&self, x_list: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mut preds = Vec::new();
        for x in x_list {
            let mut sum = 0.0;
            for i in 0..self.alpha.len() {
                sum += self.alpha[i] * self.support_labels[i] * self.kernel_fn(&self.support_vectors[i], &x);
            }
            sum += self.bias;
            preds.push(if sum >= 0.0 { 1.0 } else { -1.0 });
        }
        Ok(preds)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

impl SVM {
    fn kernel_fn(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match &self.kernel {
            SVMKernelType::Linear => dot(x1, x2),
            SVMKernelType::Polynomial { degree } => (1.0 + dot(x1, x2)).powi(*degree as i32),
            SVMKernelType::RBF { gamma } => (-*gamma * squared_distance(x1, x2)).exp(),
        }
    }
}

fn to_csc(matrix: &Vec<Vec<f64>>) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let n = matrix.len();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for j in 0..n {
        for i in 0..n {
            let v = matrix[i][j];
            if v.abs() > 1e-12 {
                data.push(v);
                indices.push(i as i32);
            }
        }
        indptr.push(data.len() as i32);
    }
    (data, indices, indptr)
}
