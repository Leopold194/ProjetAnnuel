
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
                    .map(|label| if *label == first_label { 1.0 } else { -1.0 })
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
                        .map(|v| if *v == uniq[1] { 1.0 } else { -1.0 })
                        .collect()
                } else {
                    values.clone()
                }
            }
        };

        self.label_map_str = label_map_str;
        self.label_map_float = label_map_float;
        
        for _ in 0..epochs {
            let mut rng = rand::rng();
            let i = rng.random_range(0..self.x.len());

            let random_x = self.x[i].clone();

            let prediction = self.predict_classification(random_x.clone());
            let error = labels_norm[i] - prediction;

            for j in 0..self.weights.len() - 1 {
                self.weights[j] += learning_rate * error * random_x[j];
            }
            let last_index = self.weights.len() - 1;
            self.weights[last_index] += learning_rate * error;
        }

        // println!("Weights: {:?}", self.weights);
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

        println!("X: {:?}", x);
        println!("Y: {:?}", y);

        let xt: Vec<Vec<f64>> = utils::transpose(&x);

        let xtx: Vec<Vec<f64>> = utils::matmatmul(&xt, &x);
        let call: &str = "Moore-Penrose";
        let xtx_inv: Vec<Vec<f64>> = utils::inverse(xtx.clone(), &call);
        let xtx_inv_xt: Vec<Vec<f64>> = utils::matmatmul(&xtx_inv, &xt);
        self.weights = utils::matvecmul(&xtx_inv_xt, &y);
        
        println!("Weights: {:?}", self.weights);
    }

    /// Predicts the output for a given input vector `x` using the trained model.
    /// The function calculates the weighted sum of the input features and applies the activation function.
    fn predict_classification(&self, x: Vec<f64>) -> f64 {
        let mut sum = 0.0;
        let mut x_with_bias = x.clone();
        x_with_bias.push(1.0);

        for i in 0..x_with_bias.len() {
            sum += x_with_bias[i] * self.weights[i];
        }

        if sum > 0.0 {
            1.0
        } else {
            -1.0
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