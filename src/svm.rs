use pyo3::prelude::*;
use osqp::{CscMatrix, Problem, Settings};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use serde::{Serialize, Deserialize};
#[pyclass]
#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub enum SVMKernelType {
    Linear(),
    Polynomial { degree: usize },
    RBF { gamma: f64 },
}

#[pymethods]
impl SVMKernelType {
    #[staticmethod]
    pub fn linear() -> Self {
        SVMKernelType::Linear()
    }

    #[staticmethod]
    pub fn polynomial(degree: usize) -> PyResult<Self> {
        if degree == 0 {
            Err(PyValueError::new_err("degree must be > 0"))
        } else {
            Ok(SVMKernelType::Polynomial { degree })
        }
    }

    #[staticmethod]
    pub fn rbf(gamma: f64) -> PyResult<Self> {
        if gamma <= 0.0 {
            Err(PyValueError::new_err("gamma must be > 0"))
        } else {
            Ok(SVMKernelType::RBF { gamma })
        }
    }

    fn __repr__(&self) -> String {
        match self {
            SVMKernelType::Linear() => "SVMKernelType.Linear".to_string(),
            SVMKernelType::Polynomial { degree } => format!("SVMKernelType.Polynomial(degree={})", degree),
            SVMKernelType::RBF { gamma } => format!("SVMKernelType.RBF(gamma={})", gamma),
        }
    }
}


#[pyclass]
#[derive(Serialize, Deserialize)]
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

    //matrice
    let mut q_dense = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let k = match &self.kernel {
                SVMKernelType::Linear() => dot(&x_py[i], &x_py[j]),
                SVMKernelType::Polynomial { degree } => {
                    (1.0 + dot(&x_py[i], &x_py[j])).powi(*degree as i32)
                }
                SVMKernelType::RBF { gamma } => {
                    let dist2 = squared_distance(&x_py[i], &x_py[j]);
                    (-gamma * dist2).exp()
                }
            };
            q_dense[i][j] = y_py[i] * y_py[j] * k;
            if i != j {
                q_dense[j][i] = q_dense[i][j];
            }
        }
    }

    let q_matrix = dense_to_upper_csc(q_dense);

    let q_vec = vec![-1.0; n];

    //contraintes
    let mut a_dense = vec![vec![0.0; n]; n + 1];
    for j in 0..n {
        a_dense[0][j] = y_py[j];     
        a_dense[j + 1][j] = 1.0;     
    }

    let mut a_elements = Vec::with_capacity(n * (n + 1));
    for j in 0..n {
        for i in 0..n + 1 {
            a_elements.push(a_dense[i][j]);
        }
    }

    let a = CscMatrix::from_column_iter_dense(n + 1, n, a_elements);
    
    
    let l = vec![0.0; n + 1];
    let mut u = vec![0.0; n + 1];
    for i in 1..=n {
        u[i] = std::f64::INFINITY;
    }

    // solveur quadratique
    let settings = Settings::default();
    let mut prob = Problem::new(&q_matrix, &q_vec, &a, &l, &u, &settings)
    .map_err(|e| PyRuntimeError::new_err(format!("OSQP setup error: {}", e)))?;
    let result = prob.solve();
    let alpha = result.x().unwrap().to_vec();

    // hehe c pour ca que ca s'appelle svm 
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

    // bias
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

    //faudrait que je fasse une diff entre mono et plusieurs, j'imagine une enum
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

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: SVM = serde_json::from_reader(file).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }
}

pub fn dense_to_upper_csc(dense_matrix: Vec<Vec<f64>>) -> CscMatrix<'static> {
    let n = dense_matrix.len();
    //en ft le trick c'est de stocker uniquement les valeurs trigonales sup.
    // il faut pas stocker une matrice carrée, sinon ca fait panic 
    
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    
    indptr.push(0);
    for col in 0..n {
        for row in 0..=col {  
            let val = dense_matrix[row][col];
            if val != 0.0 {    
                indices.push(row);
                data.push(val);
            }
        }
        indptr.push(data.len());
    }
    
    CscMatrix {
        nrows: n,
        ncols: n,
        indptr: indptr.into(),
        indices: indices.into(),
        data: data.into(),
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
            SVMKernelType::Linear() => dot(x1, x2),
            SVMKernelType::Polynomial { degree } => (1.0 + dot(x1, x2)).powi(*degree as i32),
            SVMKernelType::RBF { gamma } => (-*gamma * squared_distance(x1, x2)).exp(),
        }
    }
}

