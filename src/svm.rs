    use pyo3::prelude::*;
    use osqp::{CscMatrix, Problem, Settings};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::exceptions::PyValueError;
    use pyo3::pyclass;
    use serde::{Serialize, Deserialize};
    use itertools::izip;
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
        pub margin: SoftMargin,
        #[pyo3(get)]
        pub kernel: SVMKernelType,
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
                alpha: vec![],
                support_vectors: vec![],
                support_labels: vec![],
                bias: 0.0,
                kernel,
                margin: margin.unwrap_or(SoftMargin::Hard()),
            }
        }

    fn train(&mut self,x_py: Vec<Vec<f64>>, y_py: Vec<f64>) -> PyResult<()> {
        let n = x_py.len();

        // Construction de Q (le truc le plus simple mdrrr)
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
        // régularisation diagonale au cas ou
        // je peux surement retirer mtn mais c'était une piste d'exploration de pk le osqp faisait n'imp
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
                // faudrait remettre infini j'imagine
                let big_c = 1e8;
                for i in 1..=n { u[i] = big_c }
            }
        }

        // Résolution (faut changer les paramètres pour que ca marche)
        let settings = Settings::default();
        let settings = settings.polish(true);
        let settings = settings.eps_abs(1e-8);
        let settings = settings.eps_rel(1e-8);
        let settings = settings.max_iter(200_000);
        let mut prob = Problem::new(&q_matrix, &q_vec, &a, &l, &u, &settings)
            .map_err(|e| PyRuntimeError::new_err(format!("OSQP setup error: {}", e)))?;
        let status = prob.solve();  

        // prints notebook
        Python::with_gil(|py| -> PyResult<()> {
            let builtins = py.import("builtins")?;
            builtins.call_method("print", (format!("OSQP status   = {:?}", status),), None)?;
            builtins.call_method("print", (format!("Iterations    = {}", status.iter()),), None)?;
            builtins.call_method("print", (format!("Solve time    = {:?}", status.solve_time()),), None)?;
            Ok(())
        })?;

        // alpha 
        let alpha_full: &[f64] = if let Some(sol) = status.solution() {
            sol.x()
        } else {
            return Err(PyRuntimeError::new_err(
                "OSQP did not return a solved status, cannot extract alpha",
            ));
        };

        // affichage alphas
        Python::with_gil(|py| -> PyResult<()> {
            let builtins = py.import("builtins")?;
            builtins.call_method("print", (format!("alpha_full    = {:?}", alpha_full),), None)?;
            Ok(())
        })?;
        
        // vecteurs support (c'est pour ca qu'on appelle ca support vector machine)
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
        self.alpha           = sv_alpha;
        self.support_vectors = sv_x;
        self.support_labels  = sv_y;

        // Le biais
        let mut bias_sum = 0.0;
        for (ai, xi, yi) in izip!(&self.alpha, &self.support_vectors, &self.support_labels) {
            let mut s = 0.0;
            for (aj, xj, yj) in izip!(&self.alpha, &self.support_vectors, &self.support_labels) {
                s += aj * yj * self.kernel_fn(xj, xi);
            }
            bias_sum += yi - s;
        }
        self.bias = bias_sum / (self.alpha.len() as f64);

        Ok(())
    }



        //faudrait que je fasse une diff entre mono et plusieurs, j'imagine une enum pour que ca passe
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

        fn get_c(&self) -> Option<f64> {
        match self.margin {
            SoftMargin::Soft(c) => Some(c),
            SoftMargin::Hard() => None,
        }
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

