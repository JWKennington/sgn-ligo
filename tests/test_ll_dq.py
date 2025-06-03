"""Tests for sgnligo.bin.ll_dq module.

This module provides comprehensive tests for the ll_dq command-line tool,
which tracks range history for LIGO data quality monitoring.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from sgnligo.bin import ll_dq


@pytest.fixture
def mock_data_source_info():
    """Create a mock DataSourceInfo object for testing."""
    mock = Mock()
    mock.ifos = ["H1"]
    mock.data_source = "white"
    mock.input_sample_rate = 16384
    return mock


@pytest.fixture
def mock_condition_info():
    """Create a mock ConditionInfo object for testing."""
    return Mock()


@pytest.fixture
def mock_pipeline_components():
    """Create mocked pipeline components with proper return values.

    This fixture sets up all the necessary mocks for pipeline operation,
    including datasource, condition, and various tracking/sink components.
    """
    with (
        patch("sgnligo.bin.ll_dq.Pipeline") as mock_pipeline,
        patch("sgnligo.bin.ll_dq.datasource") as mock_datasource,
        patch("sgnligo.bin.ll_dq.condition") as mock_condition,
        patch("sgnligo.bin.ll_dq.HorizonDistanceTracker") as mock_tracker,
        patch("sgnligo.bin.ll_dq.NullSeriesSink") as mock_null_sink,
        patch("sgnligo.bin.ll_dq.KafkaSink") as mock_kafka_sink,
        patch("sgnligo.bin.ll_dq.HorizonDistance") as mock_horizon,
    ):

        # Setup return values for datasource and condition
        mock_datasource.return_value = ({"H1": "source_link"}, None)
        mock_condition.return_value = (
            {"H1": "condition_link"},
            {"H1": "spectrum_link"},
            None,
        )

        # Create a mock pipeline instance with a run method
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        yield {
            "pipeline": mock_pipeline,
            "pipeline_instance": mock_pipeline_instance,
            "datasource": mock_datasource,
            "condition": mock_condition,
            "tracker": mock_tracker,
            "null_sink": mock_null_sink,
            "kafka_sink": mock_kafka_sink,
            "horizon": mock_horizon,
        }


class TestParseCommandLine:
    """Test command line parsing."""

    def test_parse_command_line_minimal(self):
        """Test parsing with minimal required arguments."""
        test_args = [
            "ll_dq",
            "--data-source",
            "white",
            "--channel-name",
            "H1=FAKE-STRAIN",
            "--gps-start-time",
            "1234567890",
            "--gps-end-time",
            "1234567900",
        ]

        with patch.object(sys, "argv", test_args):
            options = ll_dq.parse_command_line()

        assert options.data_source == "white"
        assert options.channel_name == ["H1=FAKE-STRAIN"]
        assert options.gps_start_time == 1234567890
        assert options.gps_end_time == 1234567900
        assert options.output_kafka_server is None
        assert options.analysis_tag == "test"
        assert options.horizon_approximant == "IMRPhenomD"
        assert options.horizon_f_min == 15.0
        assert options.horizon_f_max == 900.0
        assert options.injections is False
        assert options.verbose is False

    def test_parse_command_line_full(self):
        """Test parsing with all optional arguments."""
        test_args = [
            "ll_dq",
            "--data-source",
            "white",
            "--channel-name",
            "H1=FAKE-STRAIN",
            "--gps-start-time",
            "1234567890",
            "--gps-end-time",
            "1234567900",
            "--output-kafka-server",
            "localhost:9092",
            "--analysis-tag",
            "mytest",
            "--horizon-approximant",
            "TaylorF2",
            "--horizon-f-min",
            "20.0",
            "--horizon-f-max",
            "1000.0",
            "--injections",
            "--verbose",
            "--whiten-sample-rate",
            "2048",
            "--psd-fft-length",
            "8",
        ]

        with patch.object(sys, "argv", test_args):
            options = ll_dq.parse_command_line()

        assert options.output_kafka_server == "localhost:9092"
        assert options.analysis_tag == "mytest"
        assert options.horizon_approximant == "TaylorF2"
        assert options.horizon_f_min == 20.0
        assert options.horizon_f_max == 1000.0
        assert options.injections is True
        assert options.verbose is True


class TestLLDQ:
    """Test the ll_dq function."""

    def test_ll_dq_single_ifo(
        self, mock_pipeline_components, mock_data_source_info, mock_condition_info
    ):
        """Test ll_dq with single IFO - the normal operating case."""
        # Call the function with test parameters
        ll_dq.ll_dq(
            data_source_info=mock_data_source_info,
            condition_info=mock_condition_info,
            output_kafka_server="localhost:9092",
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=False,
            verbose=True,
        )

        # Verify the pipeline was created and run
        mock_pipeline_components["pipeline"].assert_called_once()
        mock_pipeline_components["pipeline_instance"].run.assert_called_once()

        # Verify datasource was initialized with correct pipeline and info
        mock_pipeline_components["datasource"].assert_called_once_with(
            pipeline=mock_pipeline_components["pipeline_instance"],
            info=mock_data_source_info,
        )

        # Verify condition was called with correct parameters
        mock_pipeline_components["condition"].assert_called_once_with(
            pipeline=mock_pipeline_components["pipeline_instance"],
            condition_info=mock_condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "source_link"},
        )

        # Verify HorizonDistance was created with correct parameters
        # These are the standard NS-NS parameters (1.4 solar masses each)
        mock_pipeline_components["horizon"].assert_called_once_with(
            m1=1.4,
            m2=1.4,
            f_min=15.0,
            f_max=900.0,
            delta_f=1 / 16.0,
        )

    def test_ll_dq_multiple_ifos_raises_error(self, mock_condition_info):
        """Test that ll_dq raises error with multiple IFOs.

        The ll_dq tool currently only supports single-detector analysis.
        This test ensures proper error handling for multi-detector input.
        """
        # Create data source info with multiple IFOs
        data_source_info = Mock()
        data_source_info.ifos = ["H1", "L1"]

        with pytest.raises(ValueError, match="Only supports one ifo"):
            ll_dq.ll_dq(
                data_source_info=data_source_info,
                condition_info=mock_condition_info,
                output_kafka_server=None,
                analysis_tag="test",
                horizon_approximant="IMRPhenomD",
                horizon_f_min=15.0,
                horizon_f_max=900.0,
                injections=False,
                verbose=False,
            )

    def test_ll_dq_with_injections(
        self, mock_pipeline_components, mock_data_source_info, mock_condition_info
    ):
        """Test ll_dq with injections flag.

        When processing injection channels, the Kafka topic prefix should
        include 'inj_' to differentiate from regular data.
        """
        # Call the function with injections=True
        ll_dq.ll_dq(
            data_source_info=mock_data_source_info,
            condition_info=mock_condition_info,
            output_kafka_server="localhost:9092",
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=True,
            verbose=False,
        )

        # Verify KafkaSink was called with correct injection prefix
        kafka_sink_call = mock_pipeline_components["kafka_sink"].call_args
        assert kafka_sink_call[1]["prefix"] == "sgnl.test.inj_"

    def test_ll_dq_without_kafka_server(
        self, mock_pipeline_components, mock_data_source_info, mock_condition_info
    ):
        """Test ll_dq without kafka server (None).

        When output_kafka_server is None, the KafkaSink should be configured
        to print to stdout instead of sending to Kafka.
        """
        # Call the function with output_kafka_server=None
        ll_dq.ll_dq(
            data_source_info=mock_data_source_info,
            condition_info=mock_condition_info,
            output_kafka_server=None,
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=False,
            verbose=False,
        )

        # Verify KafkaSink was called with output_kafka_server=None
        kafka_sink_call = mock_pipeline_components["kafka_sink"].call_args
        assert kafka_sink_call[1]["output_kafka_server"] is None


class TestMain:
    """Test the main function."""

    @patch("sgnligo.bin.ll_dq.parse_command_line")
    @patch("sgnligo.bin.ll_dq.DataSourceInfo.from_options")
    @patch("sgnligo.bin.ll_dq.ConditionInfo.from_options")
    @patch("sgnligo.bin.ll_dq.ll_dq")
    def test_main(
        self,
        mock_ll_dq,
        mock_condition_from_options,
        mock_datasource_from_options,
        mock_parse_command_line,
    ):
        """Test main function."""
        # Setup mocks
        mock_options = Mock()
        mock_options.output_kafka_server = "localhost:9092"
        mock_options.analysis_tag = "test"
        mock_options.horizon_approximant = "IMRPhenomD"
        mock_options.horizon_f_min = 15.0
        mock_options.horizon_f_max = 900.0
        mock_options.injections = False
        mock_options.verbose = True

        mock_parse_command_line.return_value = mock_options

        mock_data_source_info = Mock()
        mock_datasource_from_options.return_value = mock_data_source_info

        mock_condition_info = Mock()
        mock_condition_from_options.return_value = mock_condition_info

        # Call main
        ll_dq.main()

        # Verify calls
        mock_parse_command_line.assert_called_once()
        mock_datasource_from_options.assert_called_once_with(mock_options)
        mock_condition_from_options.assert_called_once_with(mock_options)
        mock_ll_dq.assert_called_once_with(
            mock_data_source_info,
            mock_condition_info,
            "localhost:9092",
            "test",
            "IMRPhenomD",
            15.0,
            900.0,
            False,
            True,
        )

    @patch("sgnligo.bin.ll_dq.HorizonDistance")
    @patch("sgnligo.bin.ll_dq.KafkaSink")
    @patch("sgnligo.bin.ll_dq.NullSeriesSink")
    @patch("sgnligo.bin.ll_dq.HorizonDistanceTracker")
    @patch("sgnligo.bin.ll_dq.condition")
    @patch("sgnligo.bin.ll_dq.datasource")
    @patch("sgnligo.bin.ll_dq.Pipeline")
    @patch(
        "sys.argv",
        [
            "ll_dq",
            "--data-source",
            "white",
            "--channel-name",
            "H1=FAKE",
            "--gps-start-time",
            "1",
            "--gps-end-time",
            "2",
            "--input-sample-rate",
            "16384",
        ],
    )
    def test_main_entry_point(
        self, mock_pipeline, mock_ds, mock_cond, mock_hdt, mock_ns, mock_ks, mock_hd
    ):
        """Test the if __name__ == '__main__' entry point.

        This test uses runpy to execute the module as a script, ensuring that
        the entry point guard (if __name__ == '__main__') is covered. This is
        necessary for 100% coverage and verifies the script can be run directly.

        Note: The patch decorators are applied in reverse order, so mock_pipeline
        is the last patch (sys.argv) and mock_hd is the first (HorizonDistance).
        """
        # Setup minimal mocks to allow the script to run
        mock_ds.return_value = ({"H1": "link"}, None)
        mock_cond.return_value = ({"H1": "link"}, {"H1": "link"}, None)

        # Import and run the module as if it were executed directly
        import runpy

        runpy.run_module("sgnligo.bin.ll_dq", run_name="__main__")
