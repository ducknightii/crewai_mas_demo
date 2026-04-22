import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add the mock directory to the path so we can import the mock server
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mock'))

try:
    from mock_server import app
except ImportError as e:
    print(f"Error importing mock_server: {e}")
    # Create a minimal app for testing if import fails
    from flask import Flask
    app = Flask(__name__)


class TestLeaveManagementAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_get_leaves_happy_path(self):
        """Test GET /leaves - retrieve all leave records"""
        # Happy path: no filters
        response = self.app.get('/api/leaves')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIsInstance(data, list)
    
    def test_post_leaves_happy_path(self):
        """Test POST /leaves - create a new leave record"""
        # Happy path: valid leave data
        leave_data = {
            "employee_id": "EMP001",
            "employee_name": "John Doe",
            "leave_type": "annual_leave",
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "reason": "Vacation"
        }
        
        response = self.app.post('/api/leaves',
                                data=json.dumps(leave_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('id', data)
        self.assertEqual(data['employee_id'], "EMP001")
        self.assertEqual(data['status'], "pending")
    
    def test_get_leave_by_id_happy_path(self):
        """Test GET /leaves/{id} - retrieve specific leave record"""
        # First create a leave record
        leave_data = {
            "employee_id": "EMP002",
            "employee_name": "Jane Smith",
            "leave_type": "sick_leave",
            "start_date": "2024-02-01",
            "end_date": "2024-02-03",
            "reason": "Illness"
        }
        
        response = self.app.post('/api/leaves',
                                data=json.dumps(leave_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        created_data = json.loads(response.data.decode('utf-8'))
        
        # Now get the created record
        response = self.app.get(f'/api/leaves/{created_data["id"]}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['id'], created_data['id'])
    
    def test_put_leave_by_id_happy_path(self):
        """Test PUT /leaves/{id} - update existing leave record"""
        # First create a leave record
        leave_data = {
            "employee_id": "EMP003",
            "employee_name": "Bob Johnson",
            "leave_type": "personal_leave",
            "start_date": "2024-03-01",
            "end_date": "2024-03-02",
            "reason": "Family event"
        }
        
        response = self.app.post('/api/leaves',
                                data=json.dumps(leave_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        created_data = json.loads(response.data.decode('utf-8'))
        
        # Now update the record
        updated_data = {
            "employee_id": "EMP003",
            "employee_name": "Robert Johnson",
            "leave_type": "marriage_leave",
            "start_date": "2024-04-01",
            "end_date": "2024-04-10",
            "reason": "Wedding"
        }
        
        response = self.app.put(f'/api/leaves/{created_data["id"]}',
                               data=json.dumps(updated_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['employee_name'], "Robert Johnson")
        self.assertEqual(data['leave_type'], "marriage_leave")
    
    def test_delete_leave_by_id_happy_path(self):
        """Test DELETE /leaves/{id} - delete leave record"""
        # First create a leave record
        leave_data = {
            "employee_id": "EMP004",
            "employee_name": "Alice Brown",
            "leave_type": "bereavement_leave",
            "start_date": "2024-05-01",
            "end_date": "2024-05-03",
            "reason": "Family loss"
        }
        
        response = self.app.post('/api/leaves',
                                data=json.dumps(leave_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        created_data = json.loads(response.data.decode('utf-8'))
        
        # Now delete the record
        response = self.app.delete(f'/api/leaves/{created_data["id"]}')
        
        self.assertEqual(response.status_code, 204)
        
        # Verify it's gone
        response = self.app.get(f'/api/leaves/{created_data["id"]}')
        self.assertEqual(response.status_code, 404)
    
    def test_patch_leave_status_happy_path(self):
        """Test PATCH /leaves/{id}/status - update leave status"""
        # First create a leave record
        leave_data = {
            "employee_id": "EMP005",
            "employee_name": "Charlie Davis",
            "leave_type": "paternity_leave",
            "start_date": "2024-06-01",
            "end_date": "2024-06-15",
            "reason": "New baby"
        }
        
        response = self.app.post('/api/leaves',
                                data=json.dumps(leave_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        created_data = json.loads(response.data.decode('utf-8'))
        
        # Now update the status to approved
        status_data = {
            "status": "approved",
            "approver": "Manager"
        }
        
        response = self.app.patch(f'/api/leaves/{created_data["id"]}/status',
                                data=json.dumps(status_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], "approved")
        self.assertEqual(data['approver'], "Manager")


if __name__ == '__main__':
    unittest.main()