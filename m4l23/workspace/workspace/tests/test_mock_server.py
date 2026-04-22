import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add the mock directory to the path so we can import the mock server
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mock'))

try:
    from mock_server import app, leave_records
except ImportError as e:
    print(f"Error importing mock_server: {e}")
    # Create a minimal app for testing if import fails
    from flask import Flask
    app = Flask(__name__)


class TestLeaveManagementAPI(unittest.TestCase):
    """Test cases for the Employee Leave Management System API"""
    
    def setUp(self):
        """Set up test client and sample data"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Clear the in-memory storage for clean tests
        global leave_records
        leave_records.clear()
        
        # Add some initial test data
        leave_records.extend([
            {
                "id": "1",
                "employee_id": "EMP001",
                "employee_name": "张三",
                "leave_type": "annual_leave",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
                "days": 5,
                "apply_time": "2023-12-20T10:30:00.000Z",
                "status": "approved",
                "approver": "李经理",
                "reason": "年假休息"
            },
            {
                "id": "2",
                "employee_id": "EMP002",
                "employee_name": "李四",
                "leave_type": "sick_leave",
                "start_date": "2024-01-10",
                "end_date": "2024-01-12",
                "days": 3,
                "apply_time": "2024-01-08T09:15:00.000Z",
                "status": "pending",
                "approver": "",
                "reason": "感冒需要休息"
            }
        ])
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    def test_get_leaves_happy_path(self):
        """Test GET /leaves - retrieve all leave records"""
        response = self.app.get('/api/leaves')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)
    
    def test_get_leaves_with_filtering(self):
        """Test GET /leaves with query parameters for filtering"""
        # Filter by employee_name
        response = self.app.get('/api/leaves?employee_name=张三')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['employee_name'], '张三')
        
        # Filter by leave_type
        response = self.app.get('/api/leaves?leave_type=annual_leave')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['leave_type'], 'annual_leave')
        
        # Filter by status
        response = self.app.get('/api/leaves?status=pending')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['status'], 'pending')
    
    def test_create_leave_happy_path(self):
        """Test POST /leaves - create a new leave record"""
        new_leave_data = {
            "employee_id": "EMP003",
            "employee_name": "王五",
            "leave_type": "personal_leave",
            "start_date": "2024-02-01",
            "end_date": "2024-02-03",
            "reason": "处理个人事务"
        }
        
        response = self.app.post('/api/leaves',
                               data=json.dumps(new_leave_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['employee_id'], 'EMP003')
        self.assertEqual(data['employee_name'], '王五')
        self.assertEqual(data['leave_type'], 'personal_leave')
        self.assertEqual(data['start_date'], '2024-02-01')
        self.assertEqual(data['end_date'], '2024-02-03')
        self.assertEqual(data['days'], 3)
        self.assertEqual(data['status'], 'pending')
        self.assertIn('apply_time', data)
        self.assertEqual(data['reason'], '处理个人事务')
    
    def test_get_leave_by_id_happy_path(self):
        """Test GET /leaves/{id} - retrieve a specific leave record"""
        # Get the first leave record's ID
        first_leave_id = leave_records[0]['id']
        
        response = self.app.get(f'/api/leaves/{first_leave_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['id'], first_leave_id)
        self.assertEqual(data['employee_name'], '张三')
    
    def test_update_leave_happy_path(self):
        """Test PUT /leaves/{id} - update an existing leave record"""
        # Use a pending leave record for update
        pending_leave_id = leave_records[1]['id']
        
        updated_data = {
            "employee_id": "EMP002",
            "employee_name": "李四",
            "leave_type": "sick_leave",
            "start_date": "2024-01-15",
            "end_date": "2024-01-17",
            "reason": "需要更多时间恢复"
        }
        
        response = self.app.put(f'/api/leaves/{pending_leave_id}',
                              data=json.dumps(updated_data),
                              content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['id'], pending_leave_id)
        self.assertEqual(data['start_date'], '2024-01-15')
        self.assertEqual(data['end_date'], '2024-01-17')
        self.assertEqual(data['days'], 3)
        self.assertEqual(data['reason'], '需要更多时间恢复')
    
    def test_delete_leave_happy_path(self):
        """Test DELETE /leaves/{id} - delete a pending leave record"""
        # Use a pending leave record for deletion
        pending_leave_id = leave_records[1]['id']
        
        response = self.app.delete(f'/api/leaves/{pending_leave_id}')
        
        self.assertEqual(response.status_code, 204)
        
        # Verify the record is deleted
        response = self.app.get(f'/api/leaves/{pending_leave_id}')
        self.assertEqual(response.status_code, 404)
    
    def test_update_leave_status_happy_path(self):
        """Test PATCH /leaves/{id}/status - update leave status"""
        # Use a pending leave record
        pending_leave_id = leave_records[1]['id']
        
        status_update_data = {
            "status": "approved",
            "approver": "王总监"
        }
        
        response = self.app.patch(f'/api/leaves/{pending_leave_id}/status',
                                data=json.dumps(status_update_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['id'], pending_leave_id)
        self.assertEqual(data['status'], 'approved')
        self.assertEqual(data['approver'], '王总监')


if __name__ == '__main__':
    unittest.main()
