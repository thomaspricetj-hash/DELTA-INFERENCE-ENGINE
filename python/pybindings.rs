// delta_inference/python/pybindings.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::delta_f32::{compute_delta_f32, apply_delta_f32, Delta, Patch};
use crate::state_types::StateId;

/// Convert Python args (state id fields) into StateId
fn make_state_id(layer: usize, head: Option<usize>, name: Option<String>) -> StateId {
    StateId { layer, head, name }
}

/// Convert Delta -> Python dict (simple representation)
fn delta_to_py(py: Python, delta: &Delta) -> PyObject {
    let patches: Vec<(usize, f32)> = delta.patches.iter().map(|p| (p.index, p.value)).collect();
    (serde_json::to_string(&delta).unwrap_or_default()).into_py(py)
}

/// PyO3 module
#[pymodule]
fn delta_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    /// compute_delta_f32(layer, head, name, master: np.ndarray, variant: np.ndarray, tolerance: float) -> str (json)
    #[pyfn(m, "compute_delta_f32")]
    fn py_compute_delta_f32(py: Python,
        layer: usize,
        head: Option<usize>,
        name: Option<String>,
        master: PyReadonlyArray1<f32>,
        variant: PyReadonlyArray1<f32>,
        tolerance: f32
    ) -> PyResult<String> {
        let master = master.as_slice().map_err(|_| PyValueError::new_err("master not contiguous"))?;
        let variant = variant.as_slice().map_err(|_| PyValueError::new_err("variant not contiguous"))?;
        let id = make_state_id(layer, head, name);
        match compute_delta_f32(id, master, variant, tolerance) {
            Ok(delta) => Ok(serde_json::to_string(&delta).map_err(|e| PyValueError::new_err(e.to_string()))?),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    /// apply_delta_f32(master: np.ndarray, delta_json: str) -> np.ndarray
    #[pyfn(m, "apply_delta_f32")]
    fn py_apply_delta_f32(py: Python, master: PyReadonlyArray1<f32>, delta_json: String) -> PyResult<PyObject> {
        let master = master.as_slice().map_err(|_| PyValueError::new_err("master not contiguous"))?;
        let delta: Delta = serde_json::from_str(&delta_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        match apply_delta_f32(master, &delta) {
            Ok(out) => {
                let arr = PyArray1::from_vec(py, out);
                Ok(arr.into_py(py))
            }
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    Ok(())
}
