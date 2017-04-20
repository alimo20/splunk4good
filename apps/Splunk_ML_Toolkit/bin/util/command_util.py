#!/usr/bin/env python
import watchdog


def get_watchdog(time_limit=-1, memory_limit=-1, finalize_file=None):
    """Setup up 'watchdog' process to monitor resources.

    Returns:
        watchdog instance
    """
    watchdog_instance = watchdog.Watchdog(
        time_limit=time_limit,
        memory_limit=memory_limit * 1024 * 1024,
        finalize_file=finalize_file
    )
    return watchdog_instance


def is_getinfo_chunk(metadata):
    """Simply return true if the metadata action is 'getinfo'.

    Args:
        dict: metadata information from CEXC
    Returns:
        bool
    """
    return metadata['action'] == 'getinfo'
