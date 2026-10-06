import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from src.camera_events import (
    AlarmStateTracker,
    CameraAlarm,
    normalize_topic,
    parse_notifications,
)


PULL_RESPONSE = b'''<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
 xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"
 xmlns:tt="http://www.onvif.org/ver10/schema"
 xmlns:tns1="http://www.onvif.org/ver10/topics">
 <s:Body><tev:PullMessagesResponse>
  <wsnt:NotificationMessage>
   <wsnt:Topic Dialect="http://www.onvif.org/ver10/tev/topicExpression/ConcreteSet">
    tns1:UserAlarm/tns1:IVA/HumanShapeDetect</wsnt:Topic>
   <wsnt:Message><tt:Message UtcTime="1970-01-01T00:01:19Z" PropertyOperation="Changed">
    <tt:Source><tt:SimpleItem Name="VideoSourceConfigurationToken" Value="VSC0"/></tt:Source>
    <tt:Data><tt:SimpleItem Name="State" Value="true"/></tt:Data>
   </tt:Message></wsnt:Message>
  </wsnt:NotificationMessage>
  <wsnt:NotificationMessage>
   <wsnt:Topic>tns1:RuleEngine/CellMotionDetector/Motion</wsnt:Topic>
   <wsnt:Message><tt:Message UtcTime="1970-01-01T00:01:19Z" PropertyOperation="Initialized">
    <tt:Data><tt:SimpleItem Name="IsMotion" Value="false"/></tt:Data>
   </tt:Message></wsnt:Message>
  </wsnt:NotificationMessage>
 </tev:PullMessagesResponse></s:Body>
</s:Envelope>'''

T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)


def alarm(active, kind="human", at=T0, camera="cam", operation="Changed"):
    return CameraAlarm(camera, kind, active, "topic", operation, at, None)


class ParseTests(unittest.TestCase):
    def test_normalize_topic_strips_prefixes(self):
        self.assertEqual(normalize_topic(" tns1:UserAlarm/tns1:IVA/HumanShapeDetect\n"),
                         "UserAlarm/IVA/HumanShapeDetect")

    def test_parse_human_and_motion(self):
        alarms = parse_notifications(ET.fromstring(PULL_RESPONSE), "cam", T0)
        self.assertEqual([(a.kind, a.active, a.operation) for a in alarms],
                         [("human", True, "Changed"), ("motion", False, "Initialized")])
        self.assertEqual(alarms[0].camera_time, "1970-01-01T00:01:19Z")
        self.assertEqual(alarms[0].items["VideoSourceConfigurationToken"], "VSC0")

    def test_empty_response(self):
        root = ET.fromstring(b'<s:Envelope xmlns:s="http://www.w3.org/2003/05/'
                             b'soap-envelope"><s:Body/></s:Envelope>')
        self.assertEqual(parse_notifications(root, "cam"), [])


class TrackerTests(unittest.TestCase):
    def test_start_repeat_and_end(self):
        tracker = AlarmStateTracker()
        start = tracker.update(alarm(True))
        self.assertTrue(start.active)
        self.assertFalse(start.initial)
        self.assertIsNone(tracker.update(alarm(True, at=T0 + timedelta(seconds=5))))
        end = tracker.update(alarm(False, at=T0 + timedelta(seconds=20)))
        self.assertFalse(end.active)
        self.assertEqual(end.duration, 20.0)

    def test_initial_inactive_reported_once(self):
        tracker = AlarmStateTracker()
        first = tracker.update(alarm(False, operation="Initialized"))
        self.assertTrue(first.initial and not first.active)
        self.assertIsNone(tracker.update(alarm(False)))

    def test_end_without_observed_start(self):
        tracker = AlarmStateTracker()
        end = tracker.update(alarm(False))
        self.assertFalse(end.active or end.initial)
        self.assertIsNone(end.duration)

    def test_kinds_and_cameras_are_independent(self):
        tracker = AlarmStateTracker()
        tracker.update(alarm(True, kind="human"))
        self.assertIsNotNone(tracker.update(alarm(True, kind="motion")))
        self.assertIsNotNone(tracker.update(alarm(True, camera="otra")))

    def test_forget_camera_relearns_state(self):
        tracker = AlarmStateTracker()
        tracker.update(alarm(True))
        tracker.forget_camera("cam")
        again = tracker.update(alarm(True))
        self.assertTrue(again.active)

    def test_ignores_events_without_state(self):
        tracker = AlarmStateTracker()
        self.assertIsNone(tracker.update(alarm(None)))
        self.assertIsNone(tracker.update(alarm(True, kind="other")))


if __name__ == "__main__":
    unittest.main()
