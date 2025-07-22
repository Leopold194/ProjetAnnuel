use pyo3::prelude::*;
use osqp::{CscMatrix, Problem, Settings};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use serde::{Serialize, Deserialize};
use itertools::izip;
use std::collections::{HashMap, BTreeSet};

#[pyclass]
#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub enum SVMKernelType {
    Linear(),
    Polynomial { degree: usize },
    RBF { gamma: f64 },
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub enum SoftMargin {
    Hard(),
    Soft(f64),
}

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct SVM {
    model_name:String,
    #[pyo3(get)]
    pub alpha: Vec<f64>,
    #[pyo3(get)]
    pub support_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub support_labels: Vec<f64>,
    #[pyo3(get)]
    pub bias: f64,
    #[pyo3(get)]
    pub margin: SoftMargin,
    #[pyo3(get)]
    pub kernel: SVMKernelType,
}

// Une seule définition de SVMOvR
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct SVMOvR {
    model_name:String,
    classifiers: Vec<SVM>,
    classes: Vec<i32>,  // Utiliser i32 au lieu de f64 pour les classes
    kernel: SVMKernelType,   
    margin: SoftMargin,      
}

// Nouvelle structure pour One-vs-One
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct SVMOvO {
    model_name:String,
    classifiers: Vec<((i32, i32), SVM)>,
    classes: Vec<i32>,
    kernel: SVMKernelType,   
    margin: SoftMargin,
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

#[pymethods]
impl SoftMargin {
    #[staticmethod]
    pub fn hard() -> Self {
        SoftMargin::Hard()
    }

    #[staticmethod]
    pub fn soft(c: f64) -> PyResult<Self> {
        if c <= 0.0 {
            Err(PyValueError::new_err("C must be > 0"))
        } else {
            Ok(SoftMargin::Soft(c))
        }
    }

    fn __repr__(&self) -> String {
        match self {
            SoftMargin::Hard() => "SoftMargin.Hard".to_string(),
            SoftMargin::Soft(c) => format!("SoftMargin.Soft(C={})", c),
        }
    }
}

#[pymethods]
impl SVM {
    #[new]
    pub fn new(kernel: SVMKernelType, margin: Option<SoftMargin>) -> Self {
        SVM {
            model_name : String::from("SVM"),
            alpha: vec![],
            support_vectors: vec![],
            support_labels: vec![],
            bias: 0.0,
            kernel,
            margin: margin.unwrap_or(SoftMargin::Hard()),
        }
    }

    fn train(&mut self, x_py: Vec<Vec<f64>>, y_py: Vec<f64>) -> PyResult<()> {
        let n = x_py.len();

        // Construction de Q
        let mut q_dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let k = match &self.kernel {
                    SVMKernelType::Linear() => dot(&x_py[i], &x_py[j]),
                    SVMKernelType::Polynomial { degree } =>
                        (1.0 + dot(&x_py[i], &x_py[j])).powi(*degree as i32),
                    SVMKernelType::RBF { gamma } => {
                        let d2 = squared_distance(&x_py[i], &x_py[j]);
                        (-gamma * d2).exp()
                    }
                };
                q_dense[i][j] = y_py[i] * y_py[j] * k;
                q_dense[j][i] = q_dense[i][j];
            }
        }
        
        // Régularisation diagonale
        for i in 0..n {
            q_dense[i][i] += 1e-8;
        }
        
        let q_matrix = dense_to_upper_csc(q_dense);
        let q_vec = vec![-1.0; n];

        // Contraintes
        let mut a_dense = vec![vec![0.0; n]; n + 1];
        for j in 0..n {
            a_dense[0][j] = y_py[j];
            a_dense[j + 1][j] = 1.0;
        }
        let mut a_elems = Vec::with_capacity(n * (n + 1));
        for col in 0..n {
            for row in 0..=n {
                a_elems.push(a_dense[row][col]);
            }
        }
        let a = CscMatrix::from_column_iter_dense(n + 1, n, a_elems);

        let mut l = vec![0.0; n + 1];
        let mut u = vec![0.0; n + 1];
        match self.margin {
            SoftMargin::Soft(c) => {
                for i in 1..=n { u[i] = c }
            }
            SoftMargin::Hard() => {
                let big_c = 1e8;
                for i in 1..=n { u[i] = big_c }
            }
        }

        // Résolution
        let settings = Settings::default()
            .polish(true)
            .eps_abs(1e-6)
            .eps_rel(1e-6)
            .max_iter(200_000);
            
        let mut prob = Problem::new(&q_matrix, &q_vec, &a, &l, &u, &settings)
            .map_err(|e| PyRuntimeError::new_err(format!("OSQP setup error: {}", e)))?;
        let status = prob.solve();  

        // Affichage debug
        Python::with_gil(|py| -> PyResult<()> {
            let builtins = py.import("builtins")?;
            builtins.call_method1("print", (format!("OSQP status   = {:?}", status),))?;
            builtins.call_method1("print", (format!("Iterations    = {}", status.iter()),))?;
            builtins.call_method1("print", (format!("Solve time    = {:?}", status.solve_time()),))?;
            Ok(())
        })?;

        // Extraction alpha
        let alpha_full: &[f64] = if let Some(sol) = status.solution() {
            sol.x()
        } else {
            return Err(PyRuntimeError::new_err(
                "OSQP did not return a solved status, cannot extract alpha",
            ));
        };

        Python::with_gil(|py| -> PyResult<()> {
            let builtins = py.import("builtins")?;
            builtins.call_method1("print", (format!("alpha_full    = {:?}", alpha_full),))?;
            Ok(())
        })?;
        
        // Vecteurs support
        let tol = 1e-8;
        let mut sv_alpha = Vec::new();
        let mut sv_x     = Vec::new();
        let mut sv_y     = Vec::new();
        
        for (i, &ai) in alpha_full.iter().enumerate() {
            let keep = ai > tol && match self.margin {
                SoftMargin::Soft(c) => ai < c - tol,
                SoftMargin::Hard()   => true,
            };
            if keep {
                sv_alpha.push(ai);
                sv_x.push(x_py[i].clone());
                sv_y.push(y_py[i]);
            }
        }
        
        self.alpha = sv_alpha;
        self.support_vectors = sv_x;
        self.support_labels = sv_y;

        // Calcul du biais
        let mut bias_sum = 0.0;
        for (_ai, xi, yi) in izip!(&self.alpha, &self.support_vectors, &self.support_labels) {
            let mut s = 0.0;
            for (aj, xj, yj) in izip!(&self.alpha, &self.support_vectors, &self.support_labels) {
                s += aj * yj * self.kernel_fn(xj, xi);
            }
            bias_sum += yi - s;
        }
        self.bias = bias_sum / (self.alpha.len() as f64);

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

    fn is_margin_sv(&self, alpha_i: f64) -> bool {
        alpha_i > 1e-6 && match self.margin {
            SoftMargin::Soft(c) => alpha_i < c - 1e-6,
            SoftMargin::Hard() => true,
        }
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: SVM = serde_json::from_reader(file)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }

    fn get_c(&self) -> Option<f64> {
        match self.margin {
            SoftMargin::Soft(c) => Some(c),
            SoftMargin::Hard() => None,
        }
    }

    pub fn decision_function(&self, x: Vec<f64>) -> f64 {
        self.alpha.iter()
            .zip(self.support_labels.iter())
            .zip(self.support_vectors.iter())
            .map(|((&a, &y), sv)| a * y * self.kernel_fn(sv, &x))
            .sum::<f64>() + self.bias
    }
}

#[pymethods]
impl SVMOvR {
    #[new]
    pub fn new(kernel: SVMKernelType, margin: Option<SoftMargin>) -> Self {
        SVMOvR {
            model_name:String::from("SVMOvR"),
            classifiers: Vec::new(),
            classes: Vec::new(),
            kernel,
            margin: margin.unwrap_or(SoftMargin::Hard()),
        }
    }

    pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<i32>) -> PyResult<()> {
        let unique_classes: Vec<i32> = {
            let mut s = BTreeSet::new();
            for &label in &y { 
                s.insert(label); 
            }
            s.into_iter().collect()
        };

        self.classes = unique_classes.clone();
        self.classifiers = Vec::with_capacity(unique_classes.len());

        for &class_label in &self.classes {
            let binary_labels: Vec<f64> = y.iter()
                .map(|&yi| if yi == class_label { 1.0 } else { -1.0 })
                .collect();
            let mut svm = SVM::new(self.kernel.clone(), Some(self.margin.clone()));
            svm.train(x.clone(), binary_labels)?;
            self.classifiers.push(svm);
        }

        Ok(())
    
        
    }

    pub fn predict(&self, x_list: Vec<Vec<f64>>) -> PyResult<Vec<i32>> {
        let mut predictions = Vec::new();
        for x in &x_list {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_class = self.classes[0];

            for (class, svm) in self.classes.iter().zip(&self.classifiers) {
                let score = svm.decision_function(x.to_vec());
                if score > best_score {
                    best_score = score;
                    best_class = *class;
                }
            }

            predictions.push(best_class);
        }
        Ok(predictions)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: SVMOvR = serde_json::from_reader(file)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }
}

#[pymethods]
impl SVMOvO {
    #[new]
    pub fn new(kernel: SVMKernelType, margin: Option<SoftMargin>) -> Self {
        SVMOvO {
            model_name: String::from("SVMOvO"),
            classifiers: Vec::new(),
            classes: Vec::new(),
            kernel,
            margin: margin.unwrap_or(SoftMargin::Hard()),
        }
    }

    pub fn train(&mut self, x: Vec<Vec<f64>>, y: Vec<i32>) -> PyResult<()> {
        let mut classes = y.clone();
        classes.sort();
        classes.dedup();
        self.classes = classes;

        for i in 0..self.classes.len() {
            for j in i + 1..self.classes.len() {
                let class_i = self.classes[i];
                let class_j = self.classes[j];

                let mut binary_x = Vec::new();
                let mut binary_y = Vec::new();

                for (xi, &yi) in x.iter().zip(&y) {
                    if yi == class_i || yi == class_j {
                        binary_x.push(xi.clone());
                        binary_y.push(if yi == class_i { 1.0 } else { -1.0 });
                    }
                }

                let mut svm = SVM::new(self.kernel.clone(), Some(self.margin.clone()));
                svm.train(binary_x, binary_y)?;
                self.classifiers.push(((class_i, class_j), svm));
            }
        }

        Ok(())
    }

    pub fn predict(&self, x_list: Vec<Vec<f64>>) -> PyResult<Vec<i32>> {
    let mut predictions = Vec::new();

    for x in &x_list {
        let mut vote_counts = HashMap::<i32, usize>::new();

        for ((class_i, class_j), svm) in &self.classifiers {
            let score = svm.decision_function(x.to_vec());
            let winner = if score >= 0.0 { *class_i } else { *class_j };
            *vote_counts.entry(winner).or_insert(0) += 1;
        }

        let predicted = *vote_counts
            .iter()
            .max_by_key(|&(_, &v)| v)
            .unwrap()
            .0;
        predictions.push(predicted);
    }

    Ok(predictions)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        serde_json::to_writer_pretty(file, &self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let model: SVMOvO = serde_json::from_reader(file)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(model)
    }
}

// Fonctions utilitaires
pub fn dense_to_upper_csc(dense_matrix: Vec<Vec<f64>>) -> CscMatrix<'static> {
    let n = dense_matrix.len();
    
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