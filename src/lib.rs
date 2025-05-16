mod utils;
mod linear;
mod labels;
mod mlp;

use pyo3::prelude::*;
use pyo3::types::PyList;

#[allow(dead_code)]
#[pyfunction]
fn accuracy_score(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {
    let y_true = y_true.downcast_bound::<PyList>(py)?;
    let y_pred = y_pred.downcast_bound::<PyList>(py)?;

    if y_true.len() != y_pred.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_true et y_pred doivent avoir la même longueur",
        ));
    }

    let mut correct = 0;

    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        if let (Ok(list_a), Ok(list_b)) = (a.downcast::<PyList>(), b.downcast::<PyList>()) {
            if list_a.len() != list_b.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Les sous-listes doivent avoir la même taille",
                ));
            }

            let vec_a: Vec<f64> = list_a.extract()?;
            let vec_b: Vec<f64> = list_b.extract()?;

            let argmax_a = vec_a
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);
            let argmax_b = vec_b
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);

            if argmax_a == argmax_b {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<f64>(), b.extract::<f64>()) {
            if (va - vb).abs() < 1e-6 {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<i64>(), b.extract::<i64>()) {
            if va == vb {
                correct += 1;
            }
        } else if let (Ok(va), Ok(vb)) = (a.extract::<&str>(), b.extract::<&str>()) {
            if va == vb {
                correct += 1;
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Types non supportés pour la comparaison",
            ));
        }
    }

    Ok(correct as f64 / y_true.len() as f64)
}

#[allow(dead_code)]
#[pyfunction]
fn mean_squared_error(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {

    let y_true = y_true.downcast_bound::<PyList>(py)?;
    let y_pred = y_pred.downcast_bound::<PyList>(py)?;

    if y_true.len() != y_pred.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_true et y_pred doivent avoir la même longueur",
        ));
    }

    let val_nb = y_true.len();
    let mut errors = Vec::with_capacity(val_nb);

    for (a, b) in y_true.iter().zip(y_pred.iter()) {
        if let (Ok(va), Ok(vb)) = (a.extract::<f64>(), b.extract::<f64>()) {
            errors.push(va - vb);
        } else if let (Ok(va), Ok(vb)) = (a.extract::<i64>(), b.extract::<i64>()) {
            errors.push((va - vb) as f64);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Les types ne correspondent pas pour la comparaison (doivent être float ou int)",
            ));
        }
    }

    let mut squared_errors_sum: f64 = 0.0;

    for val in errors {
        squared_errors_sum += val * val;
    }

    Ok(squared_errors_sum / val_nb as f64)
}

#[allow(dead_code)]
#[pyfunction]
fn root_mean_squared_error(py: Python, y_true: PyObject, y_pred: PyObject) -> PyResult<f64> {
    let mse = mean_squared_error(py, y_true, y_pred);
    
    Ok(mse?.sqrt())
}

pub fn main() {

}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn projetannuel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accuracy_score,m)?)?;
    m.add_function(wrap_pyfunction!(mean_squared_error,m)?)?;
    m.add_function(wrap_pyfunction!(root_mean_squared_error,m)?)?;

    m.add_class::<linear::LinearModel>()?;
    m.add_class::<mlp::MLP>()?;

    m.add_function(wrap_pyfunction!(labels::float_labels,m)?)?;
    m.add_function(wrap_pyfunction!(labels::string_labels,m)?)?;

    Ok(())
}
