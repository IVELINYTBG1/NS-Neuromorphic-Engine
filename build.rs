use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let expected = manifest_dir.join(".venv-3.11").join("bin").join("python");

    let Some(pyo3_python) = std::env::var_os("PYO3_PYTHON") else {
        panic!(
            "\n\n\
             PYO3_PYTHON is not set. The binary must link against the venv's \
             libpython so brain.py's torch C-extensions match at runtime.\n\
             \n\
             Fix: `source .env && cargo build --release`\n\
             (expected PYO3_PYTHON = {})\n",
            expected.display()
        );
    };

    let got = PathBuf::from(&pyo3_python);
    let got_canon = std::fs::canonicalize(&got).unwrap_or_else(|_| got.clone());
    let want_canon = std::fs::canonicalize(&expected).unwrap_or_else(|_| expected.clone());

    if got_canon != want_canon {
        panic!(
            "\n\n\
             PYO3_PYTHON does not point at the project venv.\n\
             got:      {}\n\
             expected: {}\n\
             \n\
             Fix: `source .env && cargo build --release`\n",
            got.display(),
            expected.display()
        );
    }
}
