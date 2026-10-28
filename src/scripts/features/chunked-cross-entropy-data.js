// Generated from artifacts/reproduced-20261028T115030Z/output/report.json
// Report SHA256 79172cfdaec439ac5ad8751b2932a220f79013371857e4abba852e3167507ced; see full saved run for checks and metadata.
export const chunkedCEReportSnapshot = {
  "schema": "output-lab-v1",
  "status": "passed",
  "created_at": "2026-10-28T11:54:45.369070+00:00",
  "runtime": {
    "python": "3.12.13",
    "torch": "2.9.0",
    "numpy": "2.3.4",
    "device": "cpu",
    "dtype": "float64",
    "platform": "macOS-27.0.1-arm64-arm-64bit",
    "machine": "arm64",
    "processor": "arm",
    "threads": 1,
    "interop_threads": 1,
    "deterministic_algorithms": true
  },
  "provenance": {
    "run_id": "output",
    "report_path": "artifacts/reproduced-20261028T115030Z/output/report.json",
    "fixture_sha256": "bbcf83a557d3be98a900950c1a2dc1606bc785c7a2b933391218c4d80f8ba3e3",
    "python_sha256": "92b0d955a164e2cc47058fd174e52b99008074a8a90bc5428e8f126b867e4071",
    "correctness_report_path": "artifacts/reproduced-20261028T115030Z/output/correctness.json",
    "benchmark_seed_formula": "20261028 + N",
    "generator": "torch.Generator(device='cpu')",
    "no_downloaded_models": true,
    "report_sha256": "79172cfdaec439ac5ad8751b2932a220f79013371857e4abba852e3167507ced",
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
        "median": 0.2905835,
        "min": 0.288291,
        "max": 0.295917,
        "samples": [
          0.295667,
          0.295917,
          0.2925,
          0.29175,
          0.290709,
          0.291125,
          0.290458,
          0.289833,
          0.290416,
          0.288375,
          0.288333,
          0.288291
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
        "median": 6.732041499999999,
        "min": 6.497334,
        "max": 7.109292,
        "samples": [
          6.497334,
          7.109292,
          6.648583,
          6.661333,
          6.774875,
          6.821083,
          6.796042,
          6.686,
          6.689208,
          6.921083,
          6.665709,
          6.812625
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
        "median": 2.273646,
        "min": 2.230791,
        "max": 2.412291,
        "samples": [
          2.307042,
          2.412291,
          2.308666,
          2.246583,
          2.242583,
          2.251542,
          2.268,
          2.241084,
          2.230791,
          2.279292,
          2.407375,
          2.376666
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
        "median": 0.844854,
        "min": 0.829625,
        "max": 0.935375,
        "samples": [
          0.835042,
          0.839,
          0.85975,
          0.848333,
          0.890041,
          0.935375,
          0.86725,
          0.849875,
          0.841375,
          0.8365,
          0.841167,
          0.829625
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
        "median": 1.7495205,
        "min": 1.679875,
        "max": 1.853125,
        "samples": [
          1.744583,
          1.723166,
          1.759792,
          1.853125,
          1.754458,
          1.755417,
          1.785625,
          1.793792,
          1.732958,
          1.692375,
          1.682791,
          1.679875
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
        "median": 14.026104,
        "min": 13.685459,
        "max": 14.456291,
        "samples": [
          13.802583,
          13.948209,
          13.799,
          13.685459,
          14.0315,
          14.417625,
          14.051458,
          14.153625,
          14.020708,
          14.456291,
          13.968667,
          14.208167
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
        "median": 8.140687499999999,
        "min": 3.953333,
        "max": 13.502083,
        "samples": [
          3.963583,
          5.2295,
          13.170042,
          4.134917,
          3.953333,
          7.756167,
          9.565125,
          8.525208,
          12.718375,
          10.15975,
          5.962291,
          13.502083
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
        "median": 2.5325625,
        "min": 2.513292,
        "max": 2.676083,
        "samples": [
          2.65725,
          2.576792,
          2.676083,
          2.544042,
          2.523833,
          2.513292,
          2.571667,
          2.538708,
          2.520959,
          2.526417,
          2.518208,
          2.5175
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
      "sha256": null
    },
    {
      "url": "https://machinelearning.apple.com/research/cut-your-losses",
      "accessed": "2026-10-28",
      "read_scope": "Read title, author list, ICLR / February 2025 metadata, abstract and paper-specific GPU claims. Those performance values are not used as CPU measurements.",
      "snapshot_path": "artifacts/output-source-audit-2026-10-28/apple-research.html",
      "sha256": null
    },
    {
      "url": "https://arxiv.org/html/2411.09009v2",
      "accessed": "2026-10-28",
      "read_scope": "Read title/author/version metadata (10 March 2025), Abstract, sections 3 and 4.1-4.3, 5.2 Gradient Filtering, appendix B Removing Ignored Tokens and appendix E Floating Point Addition. Other performance tables were not independently reproduced.",
      "snapshot_path": "artifacts/output-source-audit-2026-10-28/paper.html",
      "sha256": null
    },
    {
      "url": "https://docs.pytorch.org/docs/2.9/autograd.html",
      "accessed": "2026-10-28",
      "read_scope": "Installed PyTorch 2.9.0 docstrings read for saved_tensors_hooks, Function.forward/backward/save_for_backward and F.cross_entropy. Corresponding official URL supplied for integration; online docs body not independently fetched."
    }
  ],
  "contention_note": "Other learning, failure and dynamics runs completed before this run. No independently guaranteed exclusive CPU; desktop background tasks may interfere. Single PyTorch thread; CPU timing is local to this environment.",
  "scope": "Independent small CPU H[N,D] @ W[V,D].T + optional bias, scalar valid-position mean, exact full first derivatives. No gradient filtering, tied input lookup, class weights, smoothing, higher derivatives, GPU kernels, GPU performance or actual Apple CCE package.",
  "zero_valid_policy": "Differentiable zero sentinel; mean loss is undefined and the caller must skip optimizer update. Dense reference uses sum/count for nonempty and the same explicit sentinel for empty; no NaN mean is hidden.",
  "measurement_scope": "Single-thread CPU Python/PyTorch float64 forward+backward. Each method has 3 warmups and 12 independent time samples without hooks/profiler/audit. Hooks run separately and count unique backing-storage references retained for backward; they are not allocator peaks or process RSS. Logical score tile, saved statistics, shared parameters and gradients are separate categories. No GPU speed claim. Full-sized single blocks in correctness are edge references; no-full-score observation applies to sub-full audited and benchmarked blocks."
};
