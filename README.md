## Prérequis :
- Avoir installé rust
- Avoir installé maturin (pip install maturin)
- Avoir crée et activé un environnement virtuel python (`python -m venv venv && source venv/bin/activate`)

## Compilation (test)
```bash
maturin develop
```
> A executer à chaque modification du code rust dans l'environnement virtuel python

## Compilation (release)
```bash
maturin build
pip install target/wheel/*.whl
```
> A executer pour compiler le code rust en release

# Usage
Le module installé a en python un nom correspondant a la propriété `lib.name` du fichier `Cargo.toml`
ex :
```
[package]
name = "ProjetAnnuel"
version = "0.1.0"
edition = "2021"

[lib]
name = "projetannuel"
crate-type = ["cdylib"]

[dependencies]
pyo3 = {version= "0.24.0", features = ["extension-module"] }
```

Le module python s'appellera `projetannuel` et sera importable en python avec `import projetannuel`