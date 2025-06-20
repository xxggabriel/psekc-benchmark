## Dependências:

* Um compilador C++ moderno (g++, clang++, MSVC).
* CMake (versão 3.18+).
* CUDA Toolkit (versão 11.0 ou mais recente).
* libcurl (no Linux: `sudo apt-get install libcurl4-openssl-dev`, no Windows/macOS use um gerenciador de pacotes como vcpkg ou conan).


## Compilação

```bash
# Navegue para a pasta raiz do projeto 'psekc-benchmark'

# Crie um diretório de build
mkdir build
cd build

# Execute o CMake para configurar o projeto
#    (O CMake encontrará o compilador CUDA automaticamente)
cmake ..

# Compile o projeto
cmake --build . --config Release 
# ou simplesmente 'make' em sistemas Linux/macOS

# Execute o benchmark
./benchmark
```