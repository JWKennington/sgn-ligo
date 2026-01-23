# DataSource V2 API Reference

The `datasource_v2` package provides a composable, dataclass-based API for creating gravitational wave data sources.

## Main Entry Points

### DataSource (Dispatcher)

::: sgnligo.sources.datasource_v2.DataSource
    options:
      show_root_heading: true
      members:
        - element
        - srcs
        - create_cli_parser
        - from_cli
        - list_sources
        - get_source_class

## CLI Support

::: sgnligo.sources.datasource_v2.cli
    options:
      show_root_heading: true
      members:
        - build_composed_cli_parser
        - check_composed_help_options
        - namespace_to_datasource_kwargs
        - format_composed_source_help
        - format_composed_source_list

## Registry

::: sgnligo.sources.datasource_v2.composed_registry
    options:
      show_root_heading: true
      members:
        - register_composed_source
        - get_composed_source_class
        - list_composed_source_types
        - get_composed_registry

## Source Classes

### Fake Sources

::: sgnligo.sources.datasource_v2.sources.fake
    options:
      show_root_heading: true
      members:
        - WhiteComposedSource
        - SinComposedSource
        - ImpulseComposedSource
        - WhiteRealtimeComposedSource
        - SinRealtimeComposedSource
        - ImpulseRealtimeComposedSource

### GWData Noise Sources

::: sgnligo.sources.datasource_v2.sources.gwdata_noise
    options:
      show_root_heading: true
      members:
        - GWDataNoiseComposedSource
        - GWDataNoiseRealtimeComposedSource

### Frame Sources

::: sgnligo.sources.datasource_v2.sources.frames
    options:
      show_root_heading: true
      members:
        - FramesComposedSource

### DevShm Sources

::: sgnligo.sources.datasource_v2.sources.devshm
    options:
      show_root_heading: true
      members:
        - DevShmComposedSource

### Arrakis Sources

::: sgnligo.sources.datasource_v2.sources.arrakis
    options:
      show_root_heading: true
      members:
        - ArrakisComposedSource

## Base Classes

::: sgnligo.sources.composed_base
    options:
      show_root_heading: true
      members:
        - ComposedSourceBase

## Utilities

::: sgnligo.sources.datasource_v2.sources.utils
    options:
      show_root_heading: true
      members:
        - add_state_vector_gating
