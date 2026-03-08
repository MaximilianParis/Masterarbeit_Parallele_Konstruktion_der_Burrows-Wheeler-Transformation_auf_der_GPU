# Makefile für CUDA-Projekt mit mehreren Binaries

# Compiler
NVCC := nvcc

# Flags für NVCC
NVCCFLAGS := -O2 --use_fast_math --expt-relaxed-constexpr\
             -gencode arch=compute_75,code=sm_75

# Host-Compiler-Flags
HOSTCXXFLAGS := -O2

# Quellen und Ziel-Binärnamen
SRCS := Prefix_Free_Parsing_W=5_P=16.cu \
        Prefix_Free_Parsing_W=5_P=14.cu \
        Prefix_Free_Parsing_W=5_P=13.cu \
        Prefix_Free_Parsing_W=5_P=11.cu \
        Prefix_Free_Parsing_W=5_P=23.cu \
        Prefix_Free_Parsing_W=35_P=81_small_alphabet.cu \
        Prefix_Free_Parsing_W=5_P=16_small_alphabet.cu \
        Prefix_Free_Parsing_W=5_P=23_small_alphabet.cu \
        Prefix_Free_Parsing_W=5_P=14_small_alphabet.cu \
        Lyndon_Grammar_BWT.cu \
        libcubwt.cu 
      

BINARIES := $(SRCS:.cu=_Linux)

# Standard-Buildregel
all: $(BINARIES)

# Musterregel
%_Linux: %.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(HOSTCXXFLAGS)" $< -o $@

# Aufräumen
clean:
	rm -f $(BINARIES)


