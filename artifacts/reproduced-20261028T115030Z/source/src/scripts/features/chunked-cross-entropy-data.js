// Generated from artifacts/output-lab-2026-10-28-timing-01/report.json
// Report SHA256 1cc3168ae5220100ad1fca69d3d21bd2ffa2f73c56bafa2e56ea2e1351cc404d; see full saved run for checks and metadata.
export const chunkedCEReportSnapshot = {
  "schema": "output-lab-v1",
  "status": "passed",
  "created_at": "2026-10-28T06:20:33.779071+00:00",
  "runtime": {
    "python": "3.12.14",
    "torch": "2.9.0",
    "numpy": "2.3.4",
    "device": "cpu",
    "dtype": "float64",
    "platform": "macOS-15.7.3-arm64-arm-64bit",
    "machine": "arm64",
    "processor": "arm",
    "threads": 1,
    "interop_threads": 1,
    "deterministic_algorithms": true
  },
  "provenance": {
    "run_id": "output-lab-2026-10-28-timing-01",
    "report_path": "artifacts/output-lab-2026-10-28-timing-01/report.json",
    "fixture_sha256": "8051e987d02571b2662f7ba9688b71a3f8a6165918cb6ed69cdd283d30d1de09",
    "python_sha256": "92b0d955a164e2cc47058fd174e52b99008074a8a90bc5428e8f126b867e4071",
    "correctness_report_path": "artifacts/output-lab-2026-10-28-correctness-01/correctness.json",
    "benchmark_seed_formula": "20261028 + N",
    "generator": "torch.Generator(device='cpu')",
    "no_downloaded_models": true,
    "report_sha256": "1cc3168ae5220100ad1fca69d3d21bd2ffa2f73c56bafa2e56ea2e1351cc404d",
    "extraction": "Copied actual saved run values; timings and reference counts unchanged. Per-save payload metadata omitted from this UI snapshot; full data remain in report.json."
  },
  "correctness": {
    "cases": 360,
    "comparisons": 2981,
    "max_dense_error": 4.440892098500626e-16,
    "max_js_error": 4.440892098500626e-16,
    "finite_difference_max_error": 4.6480312654306744e-11,
    "fault_cases": 12,
    "no_full_score_allocation": true,
    "no_score_saved_for_backward": true,
    "allocation_audit_scope": "Dispatch/hook audits of [N,V,D]=[5,7,3], blocks [2,2], [3,3], [5,2], [2,7]. Full-size [5,7] edge cases in the correctness sweep may form one full-sized tile and are excluded from this claim."
  },
  "measurements": [
    {
      "name": "dense-64x257x32",
      "method": "dense",
      "shape": [
        64,
        257,
        32
      ],
      "blocks": null,
      "valid_positions": 64,
      "seed": 20261092,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 0.1978335,
        "min": 0.184083,
        "max": 0.29575,
        "samples": [
          0.214334,
          0.190666,
          0.210834,
          0.186208,
          0.208375,
          0.184125,
          0.205042,
          0.186792,
          0.20225,
          0.193417,
          0.29575,
          0.184083
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 214280,
        "peak_unique_storages": 5,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 2,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 16448,
        "maximum_score_tile_elements": 16448,
        "shared_input_parameter_elements": 10529,
        "gradient_buffer_elements": 10529,
        "saved_row_statistics_elements": 0,
        "effective_labels_elements": 64,
        "score_forward_elements": 16448,
        "score_recomputed_elements": 0,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 0,
        "dH": 0,
        "dW": 0,
        "db": 0
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-64x257x32-t8-v32",
      "method": "chunked",
      "shape": [
        64,
        257,
        32
      ],
      "blocks": [
        8,
        32
      ],
      "valid_positions": 64,
      "seed": 20261092,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 4.521729499999999,
        "min": 4.463875,
        "max": 4.922291,
        "samples": [
          4.922291,
          4.69275,
          4.508417,
          4.535042,
          4.678333,
          4.463875,
          4.475125,
          4.499583,
          4.824625,
          4.501459,
          4.478292,
          4.647333
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 85768,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 16448,
        "maximum_score_tile_elements": 256,
        "shared_input_parameter_elements": 10529,
        "gradient_buffer_elements": 10529,
        "saved_row_statistics_elements": 128,
        "effective_labels_elements": 64,
        "score_forward_elements": 16448,
        "score_recomputed_elements": 16448,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 8.881784197001252e-16,
        "dH": 1.3877787807814457e-17,
        "dW": 1.3877787807814457e-17,
        "db": 5.204170427930421e-18
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-64x257x32-t16-v64",
      "method": "chunked",
      "shape": [
        64,
        257,
        32
      ],
      "blocks": [
        16,
        64
      ],
      "valid_positions": 64,
      "seed": 20261092,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 1.4904169999999999,
        "min": 1.450416,
        "max": 1.951375,
        "samples": [
          1.951375,
          1.67775,
          1.596875,
          1.496709,
          1.469209,
          1.460917,
          1.450416,
          1.477875,
          1.484125,
          1.450417,
          1.633959,
          1.51475
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 85768,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 16448,
        "maximum_score_tile_elements": 1024,
        "shared_input_parameter_elements": 10529,
        "gradient_buffer_elements": 10529,
        "saved_row_statistics_elements": 128,
        "effective_labels_elements": 64,
        "score_forward_elements": 16448,
        "score_recomputed_elements": 16448,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 0,
        "dH": 1.5612511283791264e-17,
        "dW": 1.0408340855860843e-17,
        "db": 1.734723475976807e-18
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-64x257x32-t64-v64",
      "method": "chunked",
      "shape": [
        64,
        257,
        32
      ],
      "blocks": [
        64,
        64
      ],
      "valid_positions": 64,
      "seed": 20261092,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 0.5660625,
        "min": 0.543,
        "max": 0.84825,
        "samples": [
          0.579042,
          0.552416,
          0.584042,
          0.564042,
          0.563625,
          0.570834,
          0.555375,
          0.568083,
          0.546917,
          0.543,
          0.84825,
          0.616625
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 85768,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 16448,
        "maximum_score_tile_elements": 4096,
        "shared_input_parameter_elements": 10529,
        "gradient_buffer_elements": 10529,
        "saved_row_statistics_elements": 128,
        "effective_labels_elements": 64,
        "score_forward_elements": 16448,
        "score_recomputed_elements": 16448,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 0,
        "dH": 1.5612511283791264e-17,
        "dW": 1.734723475976807e-18,
        "db": 1.734723475976807e-18
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "dense-128x1021x64",
      "method": "dense",
      "shape": [
        128,
        1021,
        64
      ],
      "blocks": null,
      "valid_positions": 128,
      "seed": 20261156,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 1.300875,
        "min": 1.059125,
        "max": 1.433583,
        "samples": [
          1.309125,
          1.390916,
          1.298167,
          1.269084,
          1.30575,
          1.366292,
          1.182875,
          1.059125,
          1.2585,
          1.433583,
          1.303583,
          1.261167
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 1634824,
        "peak_unique_storages": 5,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 2,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 130688,
        "maximum_score_tile_elements": 130688,
        "shared_input_parameter_elements": 74557,
        "gradient_buffer_elements": 74557,
        "saved_row_statistics_elements": 0,
        "effective_labels_elements": 128,
        "score_forward_elements": 130688,
        "score_recomputed_elements": 0,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 0,
        "dH": 0,
        "dW": 0,
        "db": 0
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-128x1021x64-t16-v64",
      "method": "chunked",
      "shape": [
        128,
        1021,
        64
      ],
      "blocks": [
        16,
        64
      ],
      "valid_positions": 128,
      "seed": 20261156,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 9.107271,
        "min": 8.995292,
        "max": 9.5705,
        "samples": [
          9.272709,
          9.5705,
          9.184417,
          9.336542,
          9.062833,
          9.44825,
          8.995292,
          9.069916,
          9.05075,
          9.105459,
          9.109083,
          9.071583
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 599528,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 130688,
        "maximum_score_tile_elements": 1024,
        "shared_input_parameter_elements": 74557,
        "gradient_buffer_elements": 74557,
        "saved_row_statistics_elements": 256,
        "effective_labels_elements": 128,
        "score_forward_elements": 130688,
        "score_recomputed_elements": 130688,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 8.881784197001252e-16,
        "dH": 1.1709383462843448e-17,
        "dW": 1.0408340855860843e-17,
        "db": 1.734723475976807e-18
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-128x1021x64-t32-v256",
      "method": "chunked",
      "shape": [
        128,
        1021,
        64
      ],
      "blocks": [
        32,
        256
      ],
      "valid_positions": 128,
      "seed": 20261156,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 2.6698329999999997,
        "min": 2.515542,
        "max": 2.969,
        "samples": [
          2.684208,
          2.775875,
          2.803625,
          2.655458,
          2.969,
          2.56525,
          2.519917,
          2.521208,
          2.515542,
          2.53525,
          2.831209,
          2.881625
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 599528,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 130688,
        "maximum_score_tile_elements": 8192,
        "shared_input_parameter_elements": 74557,
        "gradient_buffer_elements": 74557,
        "saved_row_statistics_elements": 256,
        "effective_labels_elements": 128,
        "score_forward_elements": 130688,
        "score_recomputed_elements": 130688,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 1.7763568394002505e-15,
        "dH": 9.107298248878237e-18,
        "dW": 1.0408340855860843e-17,
        "db": 1.734723475976807e-18
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    },
    {
      "name": "chunked-128x1021x64-t128-v256",
      "method": "chunked",
      "shape": [
        128,
        1021,
        64
      ],
      "blocks": [
        128,
        256
      ],
      "valid_positions": 128,
      "seed": 20261156,
      "warmup_runs": 3,
      "repetitions": 12,
      "wall_ms": {
        "median": 1.668958,
        "min": 1.563792,
        "max": 2.204333,
        "samples": [
          1.733375,
          1.724834,
          1.632792,
          1.643833,
          1.608334,
          1.702917,
          2.204333,
          1.595834,
          1.563792,
          1.694083,
          1.709583,
          1.627708
        ]
      },
      "saved_tensor_storage": {
        "peak_referenced_bytes": 599528,
        "peak_unique_storages": 6,
        "save_requests": 6,
        "live_bytes_after_backward": 0,
        "full_score_shape_save_requests": 0,
        "includes": "Unique backing storages referenced by tensors saved for backward, including H/W/bias and labels. Views share a storage and count once.",
        "excludes": "Unsaved tiles and GEMM temporaries, gradient buffers, allocator allocated/reserved peaks, process RSS and GPU memory."
      },
      "logical": {
        "unit": "logical elements; categories are not summed into an allocator/RSS peak",
        "dense_logits_elements": 130688,
        "maximum_score_tile_elements": 32768,
        "shared_input_parameter_elements": 74557,
        "gradient_buffer_elements": 74557,
        "saved_row_statistics_elements": 256,
        "effective_labels_elements": 128,
        "score_forward_elements": 130688,
        "score_recomputed_elements": 130688,
        "notes": "Counts refer to this implementation with all benchmark rows valid. The same score tile is overwritten for exp/p/dZ; temporary row vectors and dH/dW GEMM results also exist. H/W/bias/labels and dH/dW/db are separate persistent categories. No teaching snapshots are created."
      },
      "reference_errors": {
        "loss": 8.881784197001252e-16,
        "dH": 9.107298248878237e-18,
        "dW": 1.734723475976807e-18,
        "db": 8.673617379884035e-19
      },
      "timing_scope": "CPU forward+backward, same fixed H/W/bias/labels; includes input validation and resetting .grad=None. Excludes preallocated-input cloning, RNG, file I/O, hooks, dispatch audits and profiler. Hook sampling uses a separate call."
    }
  ],
  "sources": [
    {
      "url": "https://raw.githubusercontent.com/apple/ml-cross-entropy/main/README.md",
      "accessed": "2026-10-28",
      "read_scope": "README directly fetched and read: intro, requirements, Basic usage, implementation table with cce_exact, Generalized Usage including bias. No repository redirect claim; no fixed git commit available. Source bytes pinned by SHA256.",
      "snapshot_path": "artifacts/output-source-audit-2026-10-28/apple-cce-readme.md",
      "sha256": "cb36302c6b6c5ff1214da80fbad9f06e7b0d11f1bd3343d5a47e2aaa646ea7db"
    },
    {
      "url": "https://machinelearning.apple.com/research/cut-your-losses",
      "accessed": "2026-10-28",
      "read_scope": "Read title, author list, ICLR / February 2025 metadata, abstract and paper-specific GPU claims. Those performance values are not used as CPU measurements.",
      "snapshot_path": "artifacts/output-source-audit-2026-10-28/apple-research.html",
      "sha256": "941c8b87efdc4d468fc74b46c6ee19edd9a340edccbf189ec455616ec3c2a8d0"
    },
    {
      "url": "https://arxiv.org/html/2411.09009v2",
      "accessed": "2026-10-28",
      "read_scope": "Read title/author/version metadata (10 March 2025), Abstract, sections 3 and 4.1-4.3, 5.2 Gradient Filtering, appendix B Removing Ignored Tokens and appendix E Floating Point Addition. Other performance tables were not independently reproduced.",
      "snapshot_path": "artifacts/output-source-audit-2026-10-28/paper.html",
      "sha256": "d7100a8704e2d884a0364536131dbbee42897ab20a9794f50dd16685e60f3ee4"
    },
    {
      "url": "https://docs.pytorch.org/docs/2.9/autograd.html",
      "accessed": "2026-10-28",
      "read_scope": "Installed PyTorch 2.9.0 docstrings read for saved_tensors_hooks, Function.forward/backward/save_for_backward and F.cross_entropy. Corresponding official URL supplied for integration; online docs body not independently fetched."
    }
  ],
  "contention_note": "Parent-directed final timing slot. Immediately before sampling, read-only ps check found no learning/training_dynamics/failure/batch/dynamics/output lab processes. Other desktop/background processes may remain; no exclusive-CPU or allocator/RSS claim. One torch intra-op and inter-op thread.",
  "scope": "Independent small CPU H[N,D] @ W[V,D].T + optional bias, scalar valid-position mean, exact full first derivatives. No gradient filtering, tied input lookup, class weights, smoothing, higher derivatives, GPU kernels, GPU performance or actual Apple CCE package.",
  "zero_valid_policy": "Differentiable zero sentinel; mean loss is undefined and the caller must skip optimizer update. Dense reference uses sum/count for nonempty and the same explicit sentinel for empty; no NaN mean is hidden.",
  "measurement_scope": "Single-thread CPU Python/PyTorch float64 forward+backward. Each method has 3 warmups and 12 independent time samples without hooks/profiler/audit. Hooks run separately and count unique backing-storage references retained for backward; they are not allocator peaks or process RSS. Logical score tile, saved statistics, shared parameters and gradients are separate categories. No GPU speed claim. Full-sized single blocks in correctness are edge references; no-full-score observation applies to sub-full audited and benchmarked blocks."
};
