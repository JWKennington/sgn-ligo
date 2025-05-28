"""Tests for sgnligo.bin.ll_dq module."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from sgnligo.bin import ll_dq


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

    @patch("sgnligo.bin.ll_dq.Pipeline")
    @patch("sgnligo.bin.ll_dq.datasource")
    @patch("sgnligo.bin.ll_dq.condition")
    @patch("sgnligo.bin.ll_dq.HorizonDistanceTracker")
    @patch("sgnligo.bin.ll_dq.NullSeriesSink")
    @patch("sgnligo.bin.ll_dq.KafkaSink")
    @patch("sgnligo.bin.ll_dq.HorizonDistance")
    def test_ll_dq_single_ifo(
        self,
        mock_horizon_distance,
        mock_kafka_sink,
        mock_null_sink,
        mock_horizon_tracker,
        mock_condition,
        mock_datasource,
        mock_pipeline,
    ):
        """Test ll_dq with single IFO."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_datasource.return_value = ({"H1": "source_link"}, None)
        mock_condition.return_value = (
            {"H1": "condition_link"},
            {"H1": "spectrum_link"},
            None,
        )

        # Create data source info mock
        data_source_info = Mock()
        data_source_info.ifos = ["H1"]
        data_source_info.data_source = "white"
        data_source_info.input_sample_rate = 16384

        # Create condition info mock
        condition_info = Mock()

        # Call the function
        ll_dq.ll_dq(
            data_source_info=data_source_info,
            condition_info=condition_info,
            output_kafka_server="localhost:9092",
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=False,
            verbose=True,
        )

        # Verify calls
        mock_pipeline.assert_called_once()
        mock_datasource.assert_called_once_with(
            pipeline=mock_pipeline_instance,
            info=data_source_info,
        )
        mock_condition.assert_called_once_with(
            pipeline=mock_pipeline_instance,
            condition_info=condition_info,
            ifos=["H1"],
            data_source="white",
            input_sample_rate=16384,
            input_links={"H1": "source_link"},
        )

        # Verify HorizonDistance was created with correct parameters
        mock_horizon_distance.assert_called_once_with(
            m1=1.4,
            m2=1.4,
            f_min=15.0,
            f_max=900.0,
            delta_f=1 / 16.0,
        )

        # Verify pipeline.run was called
        mock_pipeline_instance.run.assert_called_once()

    def test_ll_dq_multiple_ifos_raises_error(self):
        """Test that ll_dq raises error with multiple IFOs."""
        # Create data source info mock with multiple IFOs
        data_source_info = Mock()
        data_source_info.ifos = ["H1", "L1"]

        condition_info = Mock()

        with pytest.raises(ValueError, match="Only supports one ifo"):
            ll_dq.ll_dq(
                data_source_info=data_source_info,
                condition_info=condition_info,
                output_kafka_server=None,
                analysis_tag="test",
                horizon_approximant="IMRPhenomD",
                horizon_f_min=15.0,
                horizon_f_max=900.0,
                injections=False,
                verbose=False,
            )

    @patch("sgnligo.bin.ll_dq.Pipeline")
    @patch("sgnligo.bin.ll_dq.datasource")
    @patch("sgnligo.bin.ll_dq.condition")
    @patch("sgnligo.bin.ll_dq.HorizonDistanceTracker")
    @patch("sgnligo.bin.ll_dq.NullSeriesSink")
    @patch("sgnligo.bin.ll_dq.KafkaSink")
    @patch("sgnligo.bin.ll_dq.HorizonDistance")
    def test_ll_dq_with_injections(
        self,
        mock_horizon_distance,
        mock_kafka_sink,
        mock_null_sink,
        mock_horizon_tracker,
        mock_condition,
        mock_datasource,
        mock_pipeline,
    ):
        """Test ll_dq with injections flag."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_datasource.return_value = ({"H1": "source_link"}, None)
        mock_condition.return_value = (
            {"H1": "condition_link"},
            {"H1": "spectrum_link"},
            None,
        )

        # Create data source info mock
        data_source_info = Mock()
        data_source_info.ifos = ["H1"]
        data_source_info.data_source = "white"
        data_source_info.input_sample_rate = 16384

        # Create condition info mock
        condition_info = Mock()

        # Call the function with injections=True
        ll_dq.ll_dq(
            data_source_info=data_source_info,
            condition_info=condition_info,
            output_kafka_server="localhost:9092",
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=True,
            verbose=False,
        )

        # Verify KafkaSink was called with correct prefix
        kafka_sink_call = mock_kafka_sink.call_args
        assert kafka_sink_call[1]["prefix"] == "sgnl.test.inj_"

    @patch("sgnligo.bin.ll_dq.Pipeline")
    @patch("sgnligo.bin.ll_dq.datasource")
    @patch("sgnligo.bin.ll_dq.condition")
    @patch("sgnligo.bin.ll_dq.HorizonDistanceTracker")
    @patch("sgnligo.bin.ll_dq.NullSeriesSink")
    @patch("sgnligo.bin.ll_dq.KafkaSink")
    @patch("sgnligo.bin.ll_dq.HorizonDistance")
    def test_ll_dq_without_kafka_server(
        self,
        mock_horizon_distance,
        mock_kafka_sink,
        mock_null_sink,
        mock_horizon_tracker,
        mock_condition,
        mock_datasource,
        mock_pipeline,
    ):
        """Test ll_dq without kafka server (None)."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_datasource.return_value = ({"H1": "source_link"}, None)
        mock_condition.return_value = (
            {"H1": "condition_link"},
            {"H1": "spectrum_link"},
            None,
        )

        # Create data source info mock
        data_source_info = Mock()
        data_source_info.ifos = ["H1"]
        data_source_info.data_source = "white"
        data_source_info.input_sample_rate = 16384

        # Create condition info mock
        condition_info = Mock()

        # Call the function with output_kafka_server=None
        ll_dq.ll_dq(
            data_source_info=data_source_info,
            condition_info=condition_info,
            output_kafka_server=None,
            analysis_tag="test",
            horizon_approximant="IMRPhenomD",
            horizon_f_min=15.0,
            horizon_f_max=900.0,
            injections=False,
            verbose=False,
        )

        # Verify KafkaSink was called with output_kafka_server=None
        kafka_sink_call = mock_kafka_sink.call_args
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

    @patch('sgnligo.bin.ll_dq.HorizonDistance')
    @patch('sgnligo.bin.ll_dq.KafkaSink')
    @patch('sgnligo.bin.ll_dq.NullSeriesSink')
    @patch('sgnligo.bin.ll_dq.HorizonDistanceTracker')
    @patch('sgnligo.bin.ll_dq.condition')
    @patch('sgnligo.bin.ll_dq.datasource')
    @patch('sgnligo.bin.ll_dq.Pipeline')
    @patch('sys.argv', ['ll_dq', '--data-source', 'white', '--channel-name', 'H1=FAKE', 
                        '--gps-start-time', '1', '--gps-end-time', '2', '--input-sample-rate', '16384'])
    def test_main_entry_point(self, mock_pipeline, mock_ds, mock_cond, mock_hdt, 
                              mock_ns, mock_ks, mock_hd):
        """Test the if __name__ == '__main__' entry point."""
        # Setup mocks
        mock_ds.return_value = ({"H1": "link"}, None)
        mock_cond.return_value = ({"H1": "link"}, {"H1": "link"}, None)
        
        # Import and run as main
        import runpy
        runpy.run_module('sgnligo.bin.ll_dq', run_name='__main__')
