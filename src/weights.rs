//! Weight loading utilities for GPT models

use anyhow::Result;
use ndarray::{Array1, Array2};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

pub struct ModelWeights {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
    pub config_json: String,
}

impl ModelWeights {
        pub fn load(path: &Path) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let data = std::fs::read(weights_file)?;
        let tensors = SafeTensors::deserialize(&data)?;
        
        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();
        
        // --- THIS IS THE CORRECTED LOOP ---
        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let dtype = view.dtype();
            let data_bytes = view.data();

            
            // 2. Deserialize the data correctly based on its type.
            let data: Vec<f32> = match dtype {
                // If it's already F32, parse it as F32.
                Dtype::F32 => {
                    // This is the safe, zero-copy, and correct way to convert a byte slice to a float slice.
                    // We are telling the compiler to trust us that these bytes represent a valid sequence of f32s.
                    let (prefix, floats, suffix) = unsafe { data_bytes.align_to::<f32>() };
                    // These asserts ensure that the memory alignment is perfect, which it should be for safetensors.
                    assert!(prefix.is_empty(), "Data was not aligned for F32");
                    assert!(suffix.is_empty(), "Data was not aligned for F32");
                    floats.to_vec()
                }
                // If it's F16 (half-precision), you must convert it.
                Dtype::F16 => {
                    // You will need the `half` crate for this.
                    // Add `half = "2.2.1"` to your Cargo.toml
                    use half::f16;
                    view.data()
                        .chunks_exact(2)
                        .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect()
                }
                // If it's BF16 (bfloat16), you must also convert it.
                Dtype::BF16 => {
                    // The `half` crate also handles bfloat16.
                    use half::bf16;
                    view.data()
                        .chunks_exact(2)
                        .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect()
                }
                // Handle other types or return an error.
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported tensor dtype {:?} for tensor '{}'",
                        dtype,
                        name
                    ));
                }
            };
            
            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }
        
        let config_file = path.join("config.json");
        let config_json = std::fs::read_to_string(config_file)?;
        
        Ok(Self {
            tensors: tensor_data,
            shapes,
            config_json,
        })
    }
    
    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(data: &[u8], config_json: &str) -> Result<Self> {
        let tensors = SafeTensors::deserialize(data)?;
        
        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();
        
        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            
            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }
        
        Ok(Self {
            tensors: tensor_data,
            shapes,
            config_json: config_json.to_string(),
        })
    }
    
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        anyhow::ensure!(
            shape.len() == 1,
            "Expected 1D tensor, got shape {:?}",
            shape
        );
        Ok(Array1::from_vec(data.clone()))
    }
    
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }
}