import unittest

from src.discovery import DEFAULT_CAMERA_MAC, parse_probe_matches


RESPONSE = b'''<?xml version="1.0"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <e:Body><d:ProbeMatches><d:ProbeMatch>
  <wsa:EndpointReference><wsa:Address>
   urn:uuid:00084633-4633-4f7f-337f-e8b723479527
  </wsa:Address></wsa:EndpointReference>
  <d:Scopes>onvif://www.onvif.org/name/W51-TY</d:Scopes>
  <d:XAddrs>http://192.168.1.51:80/onvif/device_service</d:XAddrs>
 </d:ProbeMatch></d:ProbeMatches></e:Body>
</e:Envelope>'''


class DiscoveryTests(unittest.TestCase):
    def test_probe_response_exposes_camera_host_and_identity(self):
        cameras = parse_probe_matches(RESPONSE, "192.168.1.150")
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].host, "192.168.1.51")
        self.assertEqual(cameras[0].local_address, "192.168.1.150")
        self.assertIn("e8b723479527", cameras[0].endpoint.replace("-", ""))
        self.assertEqual(DEFAULT_CAMERA_MAC, "E8:B7:23:47:95:27")

    def test_invalid_xml_is_ignored(self):
        self.assertEqual(parse_probe_matches(b"not xml", "192.0.2.1"), ())


if __name__ == "__main__":
    unittest.main()
